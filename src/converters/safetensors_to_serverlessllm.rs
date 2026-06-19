//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! Converts a directory of `*.safetensors` shards into a ServerlessLLM artifact
//! (partition files + tensor index). The pipeline is:
//! 1. Metadata scan: parse SafeTensors headers to collect tensor descriptors
//! 2. Planning: assign tensors to partitions with locality-aware balancing
//! 3. Materialization: copy tensor bytes directly from source ranges to destination offsets
//! 4. Index write: produce `tensor_index.json`
//!
//! Entry points are provided as methods on the [`SafeTensorsToServerlessLLM`] entity.

use crate::formats::error::{SaveError, SaveResult};
use crate::formats::safetensors::ids::ShardId;
use crate::formats::serverlessllm::checkpoint::Checkpoint as SllmCheckpoint;
use crate::formats::serverlessllm::ids::{PartitionCount, PartitionId};
use crate::formats::serverlessllm::tensor::TensorEntry;
use crate::formats::tensor::Dtype;
#[cfg(target_os = "linux")]
use crate::io::availability::{IoCapabilities, IoKind};
use crate::io::sync::Sync;
use crate::io::tokio::Tokio;
use crate::io::{AsyncIo, BlockingIo, ByteRange, WriteSlices};
use futures::future::try_join_all;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public API — converter entity
// ---------------------------------------------------------------------------

/// Storage engine preference for conversion operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionEnginePreference {
    /// Let the library choose based on workload characteristics.
    Adaptive,
    /// Use synchronous I/O.
    Sync,
    /// Use Tokio async I/O.
    Tokio,
    /// Use Linux io_uring (Linux only).
    #[cfg(target_os = "linux")]
    IoUring,
}

/// Orchestrates conversion from SafeTensors shards to ServerlessLLM format.
///
/// This is a configurable value object. Create it with `new()`, then optionally
/// configure with builder methods, then call conversion methods.
#[derive(Debug, Clone)]
pub struct SafeTensorsToServerlessLLM {
    input_dir: PathBuf,
    output_dir: PathBuf,
    partition_count: PartitionCount,
    engine_preference: ConversionEnginePreference,
}

impl SafeTensorsToServerlessLLM {
    /// Create a new converter configuration.
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if `partition_count` is zero.
    pub fn new(
        input_dir: impl Into<PathBuf>,
        output_dir: impl Into<PathBuf>,
        partition_count: usize,
    ) -> SaveResult<Self> {
        let count = PartitionCount::new(partition_count).ok_or_else(|| {
            SaveError::InvalidInput("partition_count must be greater than zero".to_owned())
        })?;

        Ok(Self {
            input_dir: input_dir.into(),
            output_dir: output_dir.into(),
            partition_count: count,
            engine_preference: ConversionEnginePreference::Adaptive,
        })
    }

    /// Set the engine preference (builder pattern).
    #[must_use]
    pub fn with_engine(mut self, engine: ConversionEnginePreference) -> Self {
        self.engine_preference = engine;
        self
    }

    /// Set the partition count (builder pattern).
    ///
    /// # Errors
    ///
    /// Returns `SaveError::InvalidInput` if `count` is zero.
    pub fn with_partition_count(mut self, count: usize) -> SaveResult<Self> {
        self.partition_count = PartitionCount::new(count).ok_or_else(|| {
            SaveError::InvalidInput("partition_count must be greater than zero".to_owned())
        })?;
        Ok(self)
    }

    /// Returns the configured input directory.
    #[inline]
    #[must_use]
    pub fn input_dir(&self) -> &Path {
        &self.input_dir
    }

    /// Returns the configured output directory.
    #[inline]
    #[must_use]
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Returns the configured partition count.
    #[inline]
    #[must_use]
    pub fn partition_count(&self) -> PartitionCount {
        self.partition_count
    }

    /// Returns the configured engine preference.
    #[inline]
    #[must_use]
    pub fn engine_preference(&self) -> ConversionEnginePreference {
        self.engine_preference
    }

    /// Build a conversion plan without executing it.
    pub fn plan(&self) -> SaveResult<ConversionPlan> {
        ConversionPlan::build_sync(&self.input_dir, self.partition_count.as_usize())
    }

    /// Build a conversion plan asynchronously.
    pub async fn plan_async(&self) -> SaveResult<ConversionPlan> {
        ConversionPlan::build(&self.input_dir, self.partition_count.as_usize()).await
    }

    /// Execute conversion with the configured settings.
    pub fn convert_sync(&self) -> SaveResult<()> {
        let plan = self.plan()?;
        // The sync path doesn't use engine selection - it always uses parallel sync I/O
        plan.materialize_sync(&self.output_dir)
    }

    /// Execute conversion asynchronously with the configured settings.
    pub async fn convert_async(&self) -> SaveResult<()> {
        let plan = self.plan_async().await?;
        let engine = match self.engine_preference {
            ConversionEnginePreference::Adaptive => plan.stats.choose_engine(),
            ConversionEnginePreference::Sync => ConversionEngine::Sync,
            ConversionEnginePreference::Tokio => ConversionEngine::TokioAsync,
            #[cfg(target_os = "linux")]
            ConversionEnginePreference::IoUring => ConversionEngine::IoUring,
        };
        plan.materialize(&self.output_dir, engine).await
    }

    /// Convert using adaptive storage-engine choice.
    #[deprecated(
        since = "0.1.0",
        note = "Use the entity API: SafeTensorsToServerlessLLM::new(input_dir, output_dir, count)?.convert_async().await"
    )]
    #[inline]
    pub async fn convert(
        input_dir: &str,
        output_dir: &str,
        partition_count: usize,
    ) -> SaveResult<()> {
        Self::new(input_dir, output_dir, partition_count)?
            .convert_async()
            .await
    }

    /// Convert using synchronous I/O.
    #[deprecated(
        since = "0.1.0",
        note = "Use the entity API: SafeTensorsToServerlessLLM::new(input_dir, output_dir, count)?.with_engine(ConversionEnginePreference::Sync).convert_sync()"
    )]
    #[inline]
    pub fn convert_static(
        input_dir: &str,
        output_dir: &str,
        partition_count: usize,
    ) -> SaveResult<()> {
        Self::new(input_dir, output_dir, partition_count)?
            .with_engine(ConversionEnginePreference::Sync)
            .convert_sync()
    }

    /// Convert using Tokio async I/O.
    #[deprecated(
        since = "0.1.0",
        note = "Use the entity API: SafeTensorsToServerlessLLM::new(input_dir, output_dir, count)?.with_engine(ConversionEnginePreference::Tokio).convert_async().await"
    )]
    #[inline]
    pub async fn convert_static_async(
        input_dir: &str,
        output_dir: &str,
        partition_count: usize,
    ) -> SaveResult<()> {
        Self::new(input_dir, output_dir, partition_count)?
            .with_engine(ConversionEnginePreference::Tokio)
            .convert_async()
            .await
    }

    /// Convert using Linux io_uring I/O.
    #[cfg(target_os = "linux")]
    #[deprecated(
        since = "0.1.0",
        note = "Use the entity API: SafeTensorsToServerlessLLM::new(input_dir, output_dir, count)?.with_engine(ConversionEnginePreference::IoUring).convert_sync()"
    )]
    #[inline]
    pub fn convert_static_io_uring(
        input_dir: &str,
        output_dir: &str,
        partition_count: usize,
    ) -> SaveResult<()> {
        Self::new(input_dir, output_dir, partition_count)?
            .with_engine(ConversionEnginePreference::IoUring)
            .convert_sync()
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Describes a single tensor in its source shard.
#[derive(Debug, Clone)]
pub struct TensorSource {
    pub name: String,
    pub shard_id: ShardId,
    pub shard_path: PathBuf,
    pub source_offset: u64,
    pub size: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub dtype: Dtype,
}

impl TensorSource {
    fn map_dtype(dtype: safetensors::Dtype) -> SaveResult<Dtype> {
        Ok(Dtype::from(dtype))
    }

    fn contiguous_stride(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return Vec::new();
        }
        let mut stride = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            let next_i = i + 1;
            let next_stride = stride.get(next_i).copied().unwrap_or(1);
            let next_shape = shape.get(next_i).copied().unwrap_or(1);
            stride[i] = next_stride * next_shape;
        }
        stride
    }
}

/// A single copy operation from source range to destination range.
#[derive(Debug, Clone)]
pub struct CopyOp {
    pub shard_id: ShardId,
    pub shard_path: PathBuf,
    pub source_offset: u64,
    pub dest_partition: PartitionId,
    pub dest_offset: u64,
    pub size: usize,
}

/// Statistics about the conversion plan, used for storage-engine selection.
#[derive(Debug, Clone)]
pub struct ConversionStats {
    pub total_bytes: u64,
    pub shard_count: usize,
    pub partition_count: usize,
    pub tensor_count: usize,
    pub max_shard_bytes: u64,
    pub mean_shard_bytes: u64,
    pub max_partition_bytes: u64,
    pub mean_partition_bytes: u64,
    pub mean_shards_per_partition: f64,
    pub max_shards_per_partition: usize,
    pub copy_op_count: usize,
    pub mean_copy_size: f64,
    pub max_copy_size: usize,
}

// ---------------------------------------------------------------------------
// Storage-engine selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversionEngine {
    Sync,
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
}

const LARGE_CONVERSION_THRESHOLD: u64 = 4 * 1024 * 1024 * 1024;

impl ConversionStats {
    fn choose_engine(&self) -> ConversionEngine {
        if self.total_bytes >= LARGE_CONVERSION_THRESHOLD && self.partition_count >= 4 {
            #[cfg(target_os = "linux")]
            {
                let capabilities = IoCapabilities::cached();
                if capabilities.is_available(IoKind::IoUring) {
                    return ConversionEngine::IoUring;
                }
            }
            return ConversionEngine::TokioAsync;
        }
        ConversionEngine::Sync
    }
}

// ---------------------------------------------------------------------------
// Conversion plan
// ---------------------------------------------------------------------------

/// Complete conversion plan with copy operations and index entries.
#[derive(Debug)]
pub struct ConversionPlan {
    pub copy_ops: Vec<CopyOp>,
    pub index: HashMap<String, TensorEntry>,
    pub stats: ConversionStats,
}

impl ConversionPlan {
    // -- Construction --------------------------------------------------------

    async fn build(input_dir: &Path, partition_count: usize) -> SaveResult<Self> {
        let shard_paths = Self::discover_shards(input_dir)?;
        Self::validate_index_manifest(input_dir, &shard_paths)?;
        let tensors = Self::scan_shards_mmap(&shard_paths)?;
        Self::from_tensors(tensors, partition_count)
    }

    fn build_sync(input_dir: &Path, partition_count: usize) -> SaveResult<Self> {
        let shard_paths = Self::discover_shards(input_dir)?;
        Self::validate_index_manifest(input_dir, &shard_paths)?;
        let tensors = Self::scan_shards_mmap(&shard_paths)?;
        Self::from_tensors(tensors, partition_count)
    }

    fn from_tensors(mut tensors: Vec<TensorSource>, partition_count: usize) -> SaveResult<Self> {
        if tensors.is_empty() {
            return Err(SaveError::InvalidInput(
                "no tensors found in input directory".to_owned(),
            ));
        }

        let shard_count = tensors
            .iter()
            .map(|t| t.shard_id.as_usize())
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut shard_sizes: Vec<u64> = vec![0; shard_count];
        for t in &tensors {
            shard_sizes[t.shard_id.as_usize()] += t.size as u64;
        }

        tensors.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(&b.name)));

        let mut partition_sizes: Vec<u64> = vec![0; partition_count];
        let mut partition_shards: Vec<BTreeSet<ShardId>> = vec![BTreeSet::new(); partition_count];
        let mut copy_ops: Vec<CopyOp> = Vec::with_capacity(tensors.len());
        let mut index: HashMap<String, TensorEntry> = HashMap::with_capacity(tensors.len());

        for tensor in tensors {
            let best_partition = (0..partition_count)
                .min_by_key(|&pid| {
                    let new_shard = if partition_shards[pid].contains(&tensor.shard_id) {
                        0
                    } else {
                        1
                    };
                    (partition_sizes[pid], new_shard, pid)
                })
                .unwrap_or(0);

            let offset = partition_sizes[best_partition];
            partition_sizes[best_partition] += tensor.size as u64;
            partition_shards[best_partition].insert(tensor.shard_id);

            let size_u64 = tensor.size as u64;
            let pt = TensorEntry::from_parts(
                offset,
                size_u64,
                tensor.shape.clone(),
                tensor.stride.clone(),
                tensor.dtype,
                PartitionId::new(best_partition),
            )?;
            index.insert(tensor.name.clone(), pt);

            copy_ops.push(CopyOp {
                shard_id: tensor.shard_id,
                shard_path: tensor.shard_path,
                source_offset: tensor.source_offset,
                dest_partition: PartitionId::new(best_partition),
                dest_offset: offset,
                size: tensor.size,
            });
        }

        let total_bytes: u64 = copy_ops.iter().map(|op| op.size as u64).sum();
        let max_shard_bytes = shard_sizes.iter().copied().max().unwrap_or(0);
        let mean_shard_bytes = if shard_count > 0 {
            shard_sizes.iter().sum::<u64>() / shard_count as u64
        } else {
            0
        };
        let max_partition_bytes = partition_sizes.iter().copied().max().unwrap_or(0);
        let mean_partition_bytes = if partition_count > 0 {
            partition_sizes.iter().sum::<u64>() / partition_count as u64
        } else {
            0
        };

        let shards_per_partition: Vec<usize> = partition_shards.iter().map(|s| s.len()).collect();
        let mean_shards_per_partition = if partition_count > 0 {
            shards_per_partition.iter().sum::<usize>() as f64 / partition_count as f64
        } else {
            0.0
        };
        let max_shards_per_partition = shards_per_partition.into_iter().max().unwrap_or(0);

        let copy_op_count = copy_ops.len();
        let mean_copy_size = if copy_op_count > 0 {
            total_bytes as f64 / copy_op_count as f64
        } else {
            0.0
        };
        let max_copy_size = copy_ops.iter().map(|op| op.size).max().unwrap_or(0);

        let stats = ConversionStats {
            total_bytes,
            shard_count,
            partition_count,
            tensor_count: copy_op_count,
            max_shard_bytes,
            mean_shard_bytes,
            max_partition_bytes,
            mean_partition_bytes,
            mean_shards_per_partition,
            max_shards_per_partition,
            copy_op_count,
            mean_copy_size,
            max_copy_size,
        };

        Ok(ConversionPlan {
            copy_ops,
            index,
            stats,
        })
    }

    // -- Shard discovery ----------------------------------------------------

    fn discover_shards(input_dir: &Path) -> SaveResult<Vec<PathBuf>> {
        if !input_dir.is_dir() {
            return Err(SaveError::InvalidInput(format!(
                "input path is not a directory: {}",
                input_dir.display()
            )));
        }

        let mut shards = Vec::new();
        for entry in std::fs::read_dir(input_dir).map_err(SaveError::from)? {
            let entry = entry.map_err(SaveError::from)?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.ends_with(".safetensors") {
                shards.push(path);
            }
        }

        shards.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if shards.is_empty() {
            return Err(SaveError::InvalidInput(format!(
                "no .safetensors files found in {}",
                input_dir.display()
            )));
        }

        Ok(shards)
    }

    fn validate_index_manifest(input_dir: &Path, shard_paths: &[PathBuf]) -> SaveResult<()> {
        let mut index_files = Vec::new();
        for entry in std::fs::read_dir(input_dir).map_err(SaveError::from)? {
            let entry = entry.map_err(SaveError::from)?;
            let path = entry.path();
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".safetensors.index.json"))
            {
                index_files.push(path);
            }
        }

        if index_files.is_empty() {
            return Ok(());
        }

        let shard_names: BTreeSet<String> = shard_paths
            .iter()
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|s| s.to_owned())
            })
            .collect();

        for index_path in index_files {
            let bytes = std::fs::read(&index_path).map_err(SaveError::from)?;
            let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
                SaveError::InvalidInput(format!(
                    "failed to parse index manifest {}: {e}",
                    index_path.display()
                ))
            })?;
            let weight_map = json
                .get("weight_map")
                .and_then(|value| value.as_object())
                .ok_or_else(|| {
                    SaveError::InvalidInput(format!(
                        "index manifest {} is missing weight_map",
                        index_path.display()
                    ))
                })?;

            let referenced: BTreeSet<String> = weight_map
                .values()
                .filter_map(|value| value.as_str().map(|s| s.to_owned()))
                .collect();

            if referenced != shard_names {
                return Err(SaveError::InvalidInput(format!(
                    "index manifest {} does not match discovered shard set",
                    index_path.display()
                )));
            }
        }

        Ok(())
    }

    // -- Shard scanning -----------------------------------------------------

    fn scan_shards_mmap(shard_paths: &[PathBuf]) -> SaveResult<Vec<TensorSource>> {
        use memmap2::Mmap;
        use std::fs::File;

        let mut all_tensors = Vec::new();
        let mut seen_names = std::collections::BTreeSet::new();

        for (i, shard_path) in shard_paths.iter().enumerate() {
            let file = File::open(shard_path).map_err(SaveError::from)?;
            let mmap = unsafe { Mmap::map(&file).map_err(SaveError::from)? };
            let model = SafeTensors::deserialize(&mmap)
                .map_err(|e| SaveError::Io(std::io::Error::other(e.to_string())))?;

            for name in model.names() {
                if !seen_names.insert(name.to_owned()) {
                    return Err(SaveError::InvalidInput(format!(
                        "duplicate tensor name across shards: {name}"
                    )));
                }

                let tensor = model
                    .tensor(name)
                    .map_err(|e| SaveError::InvalidInput(e.to_string()))?;
                let view = tensor.data();
                let offset = view.as_ptr() as usize - mmap.as_ptr() as usize;

                all_tensors.push(TensorSource {
                    name: name.to_owned(),
                    shard_id: ShardId::new(i),
                    shard_path: shard_path.clone(),
                    source_offset: offset as u64,
                    size: view.len(),
                    shape: tensor.shape().to_vec(),
                    stride: TensorSource::contiguous_stride(tensor.shape()),
                    dtype: TensorSource::map_dtype(tensor.dtype())?,
                });
            }
        }

        Ok(all_tensors)
    }

    // -- Materialization ----------------------------------------------------

    async fn materialize(&self, output_dir: &Path, engine: ConversionEngine) -> SaveResult<()> {
        tokio::fs::create_dir_all(output_dir).await?;

        match engine {
            ConversionEngine::Sync => self.materialize_sync_parallel(output_dir)?,
            ConversionEngine::TokioAsync => self.materialize_async_inner(output_dir).await?,
            #[cfg(target_os = "linux")]
            ConversionEngine::IoUring => self.materialize_sync_parallel(output_dir)?,
        }

        let index_path = output_dir.join("tensor_index.json");
        let index_bytes = SllmCheckpoint::encode_index(&self.index)?;
        Tokio::new()
            .write_file(&index_path, &index_bytes)
            .await
            .map_err(SaveError::from)?;
        Tokio::new()
            .sync_all(&index_path)
            .await
            .map_err(SaveError::from)?;

        Ok(())
    }

    fn materialize_sync(&self, output_dir: &Path) -> SaveResult<()> {
        std::fs::create_dir_all(output_dir)?;
        self.materialize_sync_parallel(output_dir)?;
        let index_path = output_dir.join("tensor_index.json");
        let index_bytes = SllmCheckpoint::encode_index(&self.index)?;
        Sync::new()
            .write_file(&index_path, &index_bytes)
            .map_err(SaveError::from)?;
        Sync::new().sync_all(&index_path).map_err(SaveError::from)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn materialize_io_uring(&self, output_dir: &Path) -> SaveResult<()> {
        std::fs::create_dir_all(output_dir)?;
        self.materialize_sync_parallel(output_dir)?;
        let index_path = output_dir.join("tensor_index.json");
        let index_bytes = SllmCheckpoint::encode_index(&self.index)?;
        Sync::new()
            .write_file(&index_path, &index_bytes)
            .map_err(SaveError::from)?;
        Sync::new().sync_all(&index_path).map_err(SaveError::from)?;
        Ok(())
    }

    async fn materialize_async_inner(&self, output_dir: &Path) -> SaveResult<()> {
        let partitions = self.group_by_partition();
        let futs: Vec<_> = partitions
            .into_iter()
            .map(|(partition_id, ops)| {
                let dir = output_dir.to_path_buf();
                async move { Self::write_partition_async(&dir, partition_id, &ops).await }
            })
            .collect();
        try_join_all(futs).await?;
        Ok(())
    }

    fn materialize_sync_parallel(&self, output_dir: &Path) -> SaveResult<()> {
        let partitions = self.group_by_partition();
        partitions
            .into_par_iter()
            .try_for_each(|(partition_id, ops)| {
                Self::write_partition_sync(output_dir, partition_id, &ops)
            })
    }

    fn group_by_partition(&self) -> Vec<(PartitionId, Vec<&CopyOp>)> {
        let mut by_partition: HashMap<PartitionId, Vec<&CopyOp>> = HashMap::new();
        for op in &self.copy_ops {
            by_partition.entry(op.dest_partition).or_default().push(op);
        }
        by_partition.into_iter().collect()
    }

    async fn write_partition_async(
        output_dir: &Path,
        partition_id: PartitionId,
        ops: &[&CopyOp],
    ) -> SaveResult<()> {
        let path = output_dir.join(format!("tensor.data_{}", partition_id.as_usize()));
        let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();
        let engine = Tokio::new();

        // Read all source ranges, then write everything in one positioned open.
        let mut writes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            let data = engine
                .read_range(
                    &op.shard_path,
                    ByteRange::from_offset_len(op.source_offset, op.size)?,
                )
                .await
                .map_err(SaveError::from)?;
            writes.push((op.dest_offset, data.as_ref().to_vec()));
        }

        let write_slices: Vec<crate::io::WriteSlice<'_>> = writes
            .iter()
            .map(|(offset, data)| crate::io::WriteSlice::new(*offset, data))
            .collect();
        engine
            .write_positioned_file(&path, total_size, WriteSlices::new(&write_slices)?)
            .await
            .map_err(SaveError::from)?;
        engine.sync_all(&path).await.map_err(SaveError::from)
    }

    fn write_partition_sync(
        output_dir: &Path,
        partition_id: PartitionId,
        ops: &[&CopyOp],
    ) -> SaveResult<()> {
        let path = output_dir.join(format!("tensor.data_{}", partition_id.as_usize()));
        let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();
        let engine = Sync::new();

        // Read all source ranges, then write everything in one positioned open.
        let mut writes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            let data = engine
                .read_range(
                    &op.shard_path,
                    ByteRange::from_offset_len(op.source_offset, op.size)?,
                )
                .map_err(SaveError::from)?;
            writes.push((op.dest_offset, data.as_ref().to_vec()));
        }

        let write_slices: Vec<crate::io::WriteSlice<'_>> = writes
            .iter()
            .map(|(offset, data)| crate::io::WriteSlice::new(*offset, data))
            .collect();
        engine
            .write_positioned_file(&path, total_size, WriteSlices::new(&write_slices)?)
            .map_err(SaveError::from)?;
        engine.sync_all(&path).map_err(SaveError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::Backend;
    use crate::formats::traits::{Checkpoint as _, Model as _, Tensor as TensorTrait};
    use safetensors::serialize;
    use safetensors::tensor::TensorView as StTensorView;
    use tempfile::TempDir;

    fn write_shard(path: &Path, tensors: Vec<(&str, StTensorView<'_>)>) {
        let bytes = serialize(tensors, None).expect("serialize shard");
        std::fs::write(path, bytes).unwrap();
    }

    fn make_small_model_dir(tmp: &TempDir) -> PathBuf {
        let shard = tmp.path().join("model.safetensors");
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let view = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        write_shard(&shard, vec![("weight", view)]);
        tmp.path().to_path_buf()
    }

    #[test]
    fn convert_rejects_zero_partitions() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out");
        let err = SafeTensorsToServerlessLLM::new(&tmp.path(), &out, 0).unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_rejects_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out");
        let converter = SafeTensorsToServerlessLLM::new(&tmp.path(), &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_single_shard_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = make_small_model_dir(&tmp);
        let out = tmp.path().join("out");

        SafeTensorsToServerlessLLM::new(&src, &out, 2)
            .unwrap()
            .convert_sync()
            .expect("convert");

        assert!(out.exists());
        assert!(out.join("tensor_index.json").exists());
        assert!(out.join("tensor.data_0").exists());

        let index_bytes = std::fs::read(out.join("tensor_index.json")).unwrap();
        let index: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
        let tensors = index.as_object().unwrap();
        assert!(tensors.contains_key("weight"));
    }

    #[test]
    fn convert_multi_shard_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let shard1 = src.join("model-00001-of-00002.safetensors");
        let shard2 = src.join("model-00002-of-00002.safetensors");

        let data1 = vec![0u8; 16];
        let data2 = vec![1u8; 32];
        let view1 = StTensorView::new(safetensors::Dtype::F32, vec![4], &data1).unwrap();
        let view2 = StTensorView::new(safetensors::Dtype::F32, vec![8], &data2).unwrap();
        write_shard(&shard1, vec![("a", view1)]);
        write_shard(&shard2, vec![("b", view2)]);

        let out = tmp.path().join("out");
        SafeTensorsToServerlessLLM::new(&src, &out, 4)
            .unwrap()
            .convert_sync()
            .expect("convert multi-shard");

        let index_bytes = std::fs::read(out.join("tensor_index.json")).unwrap();
        let index: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
        let tensors = index.as_object().unwrap();
        assert!(tensors.contains_key("a"));
        assert!(tensors.contains_key("b"));
        assert_eq!(tensors.len(), 2);
    }

    #[test]
    fn convert_rejects_mismatched_index_manifest() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let shard = src.join("model.safetensors");
        let data = vec![0u8; 16];
        let view = StTensorView::new(safetensors::Dtype::F32, vec![4], &data).unwrap();
        write_shard(&shard, vec![("a", view)]);

        std::fs::write(
            src.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"other.safetensors"}}"#,
        )
        .unwrap();

        let out = tmp.path().join("out");
        let converter = SafeTensorsToServerlessLLM::new(&src, &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();

        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_single_shard_roundtrip_loads_same_tensor_content() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src_roundtrip_single");
        std::fs::create_dir_all(&src).unwrap();
        let shard = src.join("model.safetensors");

        let a = vec![1u8, 2, 3, 4];
        let b = vec![5u8, 6, 7, 8, 9, 10, 11, 12];
        let view_a = StTensorView::new(safetensors::Dtype::U8, vec![4], &a).unwrap();
        let view_b = StTensorView::new(safetensors::Dtype::F32, vec![2], &b).unwrap();
        write_shard(&shard, vec![("a", view_a), ("b", view_b)]);

        let out = tmp.path().join("out_roundtrip_single");
        SafeTensorsToServerlessLLM::new(&src, &out, 2)
            .unwrap()
            .convert_sync()
            .unwrap();

        let converted =
            crate::formats::serverlessllm::Checkpoint::load(&out, Backend::Sync).unwrap();
        let a_tensor = converted.tensor("a").unwrap();
        let b_tensor = converted.tensor("b").unwrap();

        assert_eq!(a_tensor.shape(), &[4]);
        assert_eq!(a_tensor.dtype(), Dtype::U8);
        assert_eq!(a_tensor.data(), a.as_slice());

        assert_eq!(b_tensor.shape(), &[2]);
        assert_eq!(b_tensor.dtype(), Dtype::F32);
        assert_eq!(b_tensor.data(), b.as_slice());
    }

    #[test]
    fn convert_multi_shard_roundtrip_loads_same_tensor_content() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src_roundtrip_multi");
        std::fs::create_dir_all(&src).unwrap();
        let shard1 = src.join("model-00001-of-00002.safetensors");
        let shard2 = src.join("model-00002-of-00002.safetensors");

        let a = vec![1u8, 2, 3, 4];
        let b = vec![5u8, 6, 7, 8];
        let view_a = StTensorView::new(safetensors::Dtype::U8, vec![4], &a).unwrap();
        let view_b = StTensorView::new(safetensors::Dtype::U8, vec![4], &b).unwrap();
        write_shard(&shard1, vec![("a", view_a)]);
        write_shard(&shard2, vec![("b", view_b)]);

        let out = tmp.path().join("out_roundtrip_multi");
        SafeTensorsToServerlessLLM::new(&src, &out, 4)
            .unwrap()
            .convert_sync()
            .unwrap();

        let converted =
            crate::formats::serverlessllm::Checkpoint::load(&out, Backend::Sync).unwrap();
        let a_tensor = converted.tensor("a").unwrap();
        let b_tensor = converted.tensor("b").unwrap();

        assert_eq!(a_tensor.shape(), &[4]);
        assert_eq!(a_tensor.dtype(), Dtype::U8);
        assert_eq!(a_tensor.data(), a.as_slice());

        assert_eq!(b_tensor.shape(), &[4]);
        assert_eq!(b_tensor.dtype(), Dtype::U8);
        assert_eq!(b_tensor.data(), b.as_slice());
    }

    #[test]
    fn contiguous_stride_basic() {
        assert_eq!(TensorSource::contiguous_stride(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(TensorSource::contiguous_stride(&[5]), vec![1]);
        assert_eq!(TensorSource::contiguous_stride(&[]), Vec::<usize>::new());
    }

    #[test]
    fn dtype_mapping_covers_common_types() {
        assert_eq!(
            TensorSource::map_dtype(safetensors::Dtype::F32).unwrap(),
            Dtype::F32
        );
        assert_eq!(
            TensorSource::map_dtype(safetensors::Dtype::F16).unwrap(),
            Dtype::F16
        );
        assert_eq!(
            TensorSource::map_dtype(safetensors::Dtype::BF16).unwrap(),
            Dtype::Bf16
        );
        assert_eq!(
            TensorSource::map_dtype(safetensors::Dtype::I64).unwrap(),
            Dtype::I64
        );
        assert_eq!(
            TensorSource::map_dtype(safetensors::Dtype::U8).unwrap(),
            Dtype::U8
        );
    }

    #[test]
    fn dtype_mapping_exhaustive() {
        let dtypes = [
            (safetensors::Dtype::F32, Dtype::F32),
            (safetensors::Dtype::F16, Dtype::F16),
            (safetensors::Dtype::BF16, Dtype::Bf16),
            (safetensors::Dtype::F64, Dtype::F64),
            (safetensors::Dtype::I32, Dtype::I32),
            (safetensors::Dtype::I16, Dtype::I16),
            (safetensors::Dtype::I8, Dtype::I8),
            (safetensors::Dtype::I64, Dtype::I64),
            (safetensors::Dtype::U32, Dtype::U32),
            (safetensors::Dtype::U16, Dtype::U16),
            (safetensors::Dtype::U8, Dtype::U8),
            (safetensors::Dtype::U64, Dtype::U64),
            (safetensors::Dtype::BOOL, Dtype::Bool),
        ];
        for (dt, expected) in &dtypes {
            assert_eq!(TensorSource::map_dtype(*dt).unwrap(), *expected);
        }
    }

    #[test]
    fn contiguous_stride_scalar() {
        assert_eq!(TensorSource::contiguous_stride(&[1]), vec![1]);
    }

    #[test]
    fn contiguous_stride_4d() {
        assert_eq!(
            TensorSource::contiguous_stride(&[2, 3, 4, 5]),
            vec![60, 20, 5, 1]
        );
    }

    #[test]
    fn convert_async_single_shard_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = make_small_model_dir(&tmp);
        let out = tmp.path().join("out_async");

        crate::test_utils::run_async(async {
            SafeTensorsToServerlessLLM::new(&src, &out, 1)
                .unwrap()
                .convert_async()
                .await
                .expect("convert async");
        });

        assert!(out.join("tensor_index.json").exists());
    }

    #[test]
    fn convert_async_rejects_zero_partitions() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out_async_zero");

        crate::test_utils::run_async(async {
            let err = SafeTensorsToServerlessLLM::new(&tmp.path(), &out, 0).unwrap_err();
            assert!(matches!(err, SaveError::InvalidInput(_)));
        });
    }

    #[test]
    fn convert_default_engine_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = make_small_model_dir(&tmp);
        let out = tmp.path().join("out_default");

        crate::test_utils::run_async(async {
            SafeTensorsToServerlessLLM::new(&src, &out, 2)
                .unwrap()
                .convert_async()
                .await
                .expect("convert default");
        });

        assert!(out.join("tensor_index.json").exists());
    }

    #[test]
    fn engine_selection_small_single_partition() {
        let stats = ConversionStats {
            total_bytes: 1024,
            shard_count: 1,
            partition_count: 1,
            tensor_count: 1,
            max_shard_bytes: 1024,
            mean_shard_bytes: 1024,
            max_partition_bytes: 1024,
            mean_partition_bytes: 1024,
            mean_shards_per_partition: 1.0,
            max_shards_per_partition: 1,
            copy_op_count: 1,
            mean_copy_size: 1024.0,
            max_copy_size: 1024,
        };
        assert_eq!(stats.choose_engine(), ConversionEngine::Sync);
    }

    #[test]
    fn engine_selection_large_multi() {
        let stats = ConversionStats {
            total_bytes: 16 * 1024 * 1024 * 1024,
            shard_count: 8,
            partition_count: 32,
            tensor_count: 500,
            max_shard_bytes: 2 * 1024 * 1024 * 1024,
            mean_shard_bytes: 2 * 1024 * 1024 * 1024,
            max_partition_bytes: 512 * 1024 * 1024,
            mean_partition_bytes: 512 * 1024 * 1024,
            mean_shards_per_partition: 2.0,
            max_shards_per_partition: 4,
            copy_op_count: 500,
            mean_copy_size: 32.0 * 1024.0 * 1024.0,
            max_copy_size: 64 * 1024 * 1024,
        };
        assert_ne!(stats.choose_engine(), ConversionEngine::Sync);
    }

    #[test]
    fn choose_sync_for_small_conversion() {
        let stats = ConversionStats {
            total_bytes: 1024 * 1024,
            shard_count: 2,
            partition_count: 4,
            tensor_count: 10,
            max_shard_bytes: 512 * 1024,
            mean_shard_bytes: 512 * 1024,
            max_partition_bytes: 256 * 1024,
            mean_partition_bytes: 256 * 1024,
            mean_shards_per_partition: 1.0,
            max_shards_per_partition: 2,
            copy_op_count: 10,
            mean_copy_size: 100_000.0,
            max_copy_size: 200_000,
        };
        assert_eq!(stats.choose_engine(), ConversionEngine::Sync);
    }

    #[test]
    fn convert_duplicate_tensor_name_rejected() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let shard1 = src.join("model-00001-of-00002.safetensors");
        let shard2 = src.join("model-00002-of-00002.safetensors");

        let data = vec![0u8; 8];
        let view1 = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        let view2 = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        write_shard(&shard1, vec![("same_name", view1)]);
        write_shard(&shard2, vec![("same_name", view2)]);

        let out = tmp.path().join("out_dup");
        let converter = SafeTensorsToServerlessLLM::new(&src, &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn partition_balancing_many_tensors() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let shard = src.join("model.safetensors");

        let data_a = vec![0u8; 16];
        let data_b = vec![0u8; 32];
        let data_c = vec![0u8; 64];
        let data_d = vec![0u8; 128];
        let va = StTensorView::new(safetensors::Dtype::U8, vec![16], &data_a).unwrap();
        let vb = StTensorView::new(safetensors::Dtype::U8, vec![32], &data_b).unwrap();
        let vc = StTensorView::new(safetensors::Dtype::U8, vec![64], &data_c).unwrap();
        let vd = StTensorView::new(safetensors::Dtype::U8, vec![128], &data_d).unwrap();
        write_shard(&shard, vec![("a", va), ("b", vb), ("c", vc), ("d", vd)]);

        let out = tmp.path().join("out_balance");
        SafeTensorsToServerlessLLM::new(&src, &out, 2)
            .unwrap()
            .convert_sync()
            .unwrap();

        let index_bytes = std::fs::read(out.join("tensor_index.json")).unwrap();
        let index: serde_json::Value = serde_json::from_slice(&index_bytes).unwrap();
        let tensors = index.as_object().unwrap();
        assert_eq!(tensors.len(), 4);

        let partitions: std::collections::BTreeSet<u64> = tensors
            .values()
            .filter_map(|v| v.as_array().and_then(|a| a[5].as_u64()))
            .collect();
        assert!(
            partitions.len() <= 2,
            "expected at most 2 partitions, got {partitions:?}"
        );
    }

    #[test]
    fn discover_shards_sorts_paths() {
        let tmp = TempDir::new().unwrap();
        let z = tmp.path().join("z_shard.safetensors");
        let a = tmp.path().join("a_shard.safetensors");
        let data = vec![0u8; 4];
        let v = StTensorView::new(safetensors::Dtype::U8, vec![4], &data).unwrap();
        write_shard(&z, vec![("z", v.clone())]);
        let v2 = StTensorView::new(safetensors::Dtype::U8, vec![4], &data).unwrap();
        write_shard(&a, vec![("a", v2)]);

        let shards = ConversionPlan::discover_shards(tmp.path()).unwrap();
        assert!(shards[0].file_name().unwrap() < shards[1].file_name().unwrap());
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn stride_product_equals_element_count(
                ndim in 1usize..6,
                seed in any::<u64>()
            ) {
                let mut rng = seed;
                let shape: Vec<usize> = (0..ndim)
                    .map(|_| {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((rng >> 33) % 10 + 1) as usize
                    })
                    .collect();
                let stride = TensorSource::contiguous_stride(&shape);
                prop_assert_eq!(stride.len(), shape.len());
                if !shape.is_empty() {
                    let total: usize = shape.iter().product();
                    prop_assert_eq!(stride[0] * shape[0], total);
                }
            }

            #[test]
            fn stride_last_is_one(ndim in 1usize..8) {
                let shape: Vec<usize> = vec![2; ndim];
                let stride = TensorSource::contiguous_stride(&shape);
                prop_assert_eq!(*stride.last().unwrap(), 1);
            }
        }
    }
}
