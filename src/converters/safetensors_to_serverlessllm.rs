//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! Converts a directory of `*.safetensors` files into a ServerlessLLM artifact
//! (partition files + tensor index). The pipeline is:
//! 1. Metadata scan: parse SafeTensors headers to collect tensor descriptors
//! 2. Planning: assign tensors to partitions with locality-aware balancing
//! 3. Materialization: copy tensor bytes directly from source ranges to destination offsets
//! 4. Index write: produce `tensor_index.json`
//!
//! Entry points are provided as methods on the [`SafeTensorsToServerlessLLM`] entity.

use crate::formats::error::{SaveError, SaveResult};

use crate::formats::serverlessllm::checkpoint::Checkpoint as SllmCheckpoint;
use crate::formats::serverlessllm::ids::{PartitionCount, PartitionId};
use crate::formats::serverlessllm::tensor::TensorEntry;
use crate::formats::tensor::{Dtype, TensorMeta};
use fastio::{WriteSlice, WriteSlices, sync, tokio as fastio_tokio};
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
    /// Use synchronous I/O.
    Sync,
    /// Use Tokio async I/O.
    Tokio,
    /// Use Linux io_uring (Linux only).
    #[cfg(target_os = "linux")]
    IoUring,
}

/// Orchestrates conversion from a directory of SafeTensors files to ServerlessLLM format.
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
            engine_preference: ConversionEnginePreference::Sync,
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
        plan.materialize_sync(&self.output_dir)
    }

    /// Execute conversion asynchronously with the configured settings.
    pub async fn convert_async(&self) -> SaveResult<()> {
        let plan = self.plan_async().await?;
        let engine = match self.engine_preference {
            ConversionEnginePreference::Sync => ConversionEngine::Sync,
            ConversionEnginePreference::Tokio => ConversionEngine::TokioAsync,
            #[cfg(target_os = "linux")]
            ConversionEnginePreference::IoUring => ConversionEngine::IoUring,
        };
        plan.materialize(&self.output_dir, engine).await
    }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Describes a single tensor in its source SafeTensors file.
#[derive(Debug, Clone)]
pub struct TensorSource {
    pub name: String,
    /// Index of the source `.safetensors` file in discovery order.
    pub source_file_index: usize,
    pub source_path: PathBuf,
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

/// A single copy operation from a source SafeTensors file to a destination partition.
#[derive(Debug, Clone)]
pub struct CopyOp {
    /// Index of the source `.safetensors` file in discovery order.
    pub source_file_index: usize,
    pub source_path: PathBuf,
    pub source_offset: u64,
    pub dest_partition: PartitionId,
    pub dest_offset: u64,
    pub size: usize,
}

/// Statistics about the conversion plan.
#[derive(Debug, Clone)]
pub struct ConversionStats {
    pub total_bytes: u64,
    pub source_file_count: usize,
    pub partition_count: usize,
    pub tensor_count: usize,
    pub max_source_file_bytes: u64,
    pub mean_source_file_bytes: u64,
    pub max_partition_bytes: u64,
    pub mean_partition_bytes: u64,
    pub mean_source_files_per_partition: f64,
    pub max_source_files_per_partition: usize,
    pub copy_op_count: usize,
    pub mean_copy_size: f64,
    pub max_copy_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversionEngine {
    Sync,
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
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
        let source_paths = Self::discover_source_files(input_dir)?;
        Self::validate_index_manifest(input_dir, &source_paths)?;
        let tensors = Self::scan_source_files_mmap(&source_paths)?;
        Self::from_tensors(tensors, partition_count)
    }

    fn build_sync(input_dir: &Path, partition_count: usize) -> SaveResult<Self> {
        let source_paths = Self::discover_source_files(input_dir)?;
        Self::validate_index_manifest(input_dir, &source_paths)?;
        let tensors = Self::scan_source_files_mmap(&source_paths)?;
        Self::from_tensors(tensors, partition_count)
    }

    fn from_tensors(mut tensors: Vec<TensorSource>, partition_count: usize) -> SaveResult<Self> {
        if tensors.is_empty() {
            return Err(SaveError::InvalidInput(
                "no tensors found in input directory".to_owned(),
            ));
        }

        let source_file_count = tensors
            .iter()
            .map(|t| t.source_file_index)
            .max()
            .map(|m| m + 1)
            .unwrap_or(1);

        let mut source_file_sizes: Vec<u64> = vec![0; source_file_count];
        for t in &tensors {
            source_file_sizes[t.source_file_index] += t.size as u64;
        }

        tensors.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(&b.name)));

        let mut partition_sizes: Vec<u64> = vec![0; partition_count];
        let mut partition_source_files: Vec<BTreeSet<usize>> =
            vec![BTreeSet::new(); partition_count];
        let mut copy_ops: Vec<CopyOp> = Vec::with_capacity(tensors.len());
        let mut index: HashMap<String, TensorEntry> = HashMap::with_capacity(tensors.len());

        for tensor in tensors {
            let best_partition = (0..partition_count)
                .min_by_key(|&pid| {
                    let new_source_file =
                        if partition_source_files[pid].contains(&tensor.source_file_index) {
                            0
                        } else {
                            1
                        };
                    (partition_sizes[pid], new_source_file, pid)
                })
                .unwrap_or(0);

            let offset = partition_sizes[best_partition];
            partition_sizes[best_partition] += tensor.size as u64;
            partition_source_files[best_partition].insert(tensor.source_file_index);

            let size_u64 = tensor.size as u64;
            let meta = TensorMeta::new(
                offset,
                size_u64,
                tensor.shape.clone(),
                tensor.stride.clone(),
                tensor.dtype,
            )?;
            let pt = TensorEntry::new(meta, PartitionId::new(best_partition));
            index.insert(tensor.name.clone(), pt);

            copy_ops.push(CopyOp {
                source_file_index: tensor.source_file_index,
                source_path: tensor.source_path,
                source_offset: tensor.source_offset,
                dest_partition: PartitionId::new(best_partition),
                dest_offset: offset,
                size: tensor.size,
            });
        }

        let total_bytes: u64 = copy_ops.iter().map(|op| op.size as u64).sum();
        let max_source_file_bytes = source_file_sizes.iter().copied().max().unwrap_or(0);
        let mean_source_file_bytes = if source_file_count > 0 {
            source_file_sizes.iter().sum::<u64>() / source_file_count as u64
        } else {
            0
        };
        let max_partition_bytes = partition_sizes.iter().copied().max().unwrap_or(0);
        let mean_partition_bytes = if partition_count > 0 {
            partition_sizes.iter().sum::<u64>() / partition_count as u64
        } else {
            0
        };

        let source_files_per_partition: Vec<usize> =
            partition_source_files.iter().map(|s| s.len()).collect();
        let mean_source_files_per_partition = if partition_count > 0 {
            source_files_per_partition.iter().sum::<usize>() as f64 / partition_count as f64
        } else {
            0.0
        };
        let max_source_files_per_partition =
            source_files_per_partition.into_iter().max().unwrap_or(0);

        let copy_op_count = copy_ops.len();
        let mean_copy_size = if copy_op_count > 0 {
            total_bytes as f64 / copy_op_count as f64
        } else {
            0.0
        };
        let max_copy_size = copy_ops.iter().map(|op| op.size).max().unwrap_or(0);

        let stats = ConversionStats {
            total_bytes,
            source_file_count,
            partition_count,
            tensor_count: copy_op_count,
            max_source_file_bytes,
            mean_source_file_bytes,
            max_partition_bytes,
            mean_partition_bytes,
            mean_source_files_per_partition,
            max_source_files_per_partition,
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

    // -- Source file discovery -----------------------------------------------

    fn discover_source_files(input_dir: &Path) -> SaveResult<Vec<PathBuf>> {
        if !input_dir.is_dir() {
            return Err(SaveError::InvalidInput(format!(
                "input path is not a directory: {}",
                input_dir.display()
            )));
        }

        let mut source_files = Vec::new();
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
                source_files.push(path);
            }
        }

        source_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if source_files.is_empty() {
            return Err(SaveError::InvalidInput(format!(
                "no .safetensors files found in {}",
                input_dir.display()
            )));
        }

        Ok(source_files)
    }

    fn validate_index_manifest(input_dir: &Path, source_paths: &[PathBuf]) -> SaveResult<()> {
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

        let source_file_names: BTreeSet<String> = source_paths
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

            if referenced != source_file_names {
                return Err(SaveError::InvalidInput(format!(
                    "index manifest {} does not match discovered source file set",
                    index_path.display()
                )));
            }
        }

        Ok(())
    }

    // -- Source file scanning -----------------------------------------------

    fn scan_source_files_mmap(source_paths: &[PathBuf]) -> SaveResult<Vec<TensorSource>> {
        use memmap2::Mmap;
        use std::fs::File;

        let mut all_tensors = Vec::new();
        let mut seen_names = std::collections::BTreeSet::new();

        for (i, source_path) in source_paths.iter().enumerate() {
            let file = File::open(source_path).map_err(SaveError::from)?;
            let mmap = unsafe { Mmap::map(&file).map_err(SaveError::from)? };
            let model = SafeTensors::deserialize(&mmap)
                .map_err(|e| SaveError::Io(std::io::Error::other(e.to_string())))?;

            for name in model.names() {
                if !seen_names.insert(name.to_owned()) {
                    return Err(SaveError::InvalidInput(format!(
                        "duplicate tensor name across source files: {name}"
                    )));
                }

                let tensor = model
                    .tensor(name)
                    .map_err(|e| SaveError::InvalidInput(e.to_string()))?;
                let view = tensor.data();
                let offset = view.as_ptr() as usize - mmap.as_ptr() as usize;

                all_tensors.push(TensorSource {
                    name: name.to_owned(),
                    source_file_index: i,
                    source_path: source_path.clone(),
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
            ConversionEngine::IoUring => self.materialize_uring_parallel(output_dir)?,
        }

        let index_path = output_dir.join("tensor_index.json");
        let index_bytes = SllmCheckpoint::encode_index(&self.index)?;
        let index_file = fastio_tokio::File::create(&index_path)
            .await
            .map_err(SaveError::from)?;
        index_file
            .write_all_at(0, &index_bytes)
            .await
            .map_err(SaveError::from)?;
        index_file.sync_all().await.map_err(SaveError::from)?;

        Ok(())
    }

    fn materialize_sync(&self, output_dir: &Path) -> SaveResult<()> {
        std::fs::create_dir_all(output_dir)?;
        self.materialize_sync_parallel(output_dir)?;
        let index_path = output_dir.join("tensor_index.json");
        let index_bytes = SllmCheckpoint::encode_index(&self.index)?;
        let index_file = sync::File::create(&index_path).map_err(SaveError::from)?;
        index_file
            .write_all_at(0, &index_bytes)
            .map_err(SaveError::from)?;
        index_file.sync_all().map_err(SaveError::from)?;
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

    #[cfg(target_os = "linux")]
    fn materialize_uring_parallel(&self, output_dir: &Path) -> SaveResult<()> {
        let partitions = self.group_by_partition();
        partitions
            .into_par_iter()
            .try_for_each(|(partition_id, ops)| {
                Self::write_partition_uring(output_dir, partition_id, &ops)
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

        // Read all source ranges, then write everything in one positioned open.
        let mut writes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            let source = fastio_tokio::File::open(&op.source_path)
                .await
                .map_err(SaveError::from)?;
            let data = source
                .read_at(op.source_offset, op.size)
                .await
                .map_err(SaveError::from)?;
            writes.push((op.dest_offset, data.as_ref().to_vec()));
        }

        let write_slices: Vec<WriteSlice<'_>> = writes
            .iter()
            .map(|(offset, data)| WriteSlice::new(*offset, data))
            .collect();
        let output = fastio_tokio::File::create(&path)
            .await
            .map_err(SaveError::from)?;
        output.set_len(total_size).await.map_err(SaveError::from)?;
        output
            .write_slices_at(WriteSlices::new(&write_slices).map_err(SaveError::from)?)
            .await
            .map_err(SaveError::from)?;
        output.sync_all().await.map_err(SaveError::from)
    }

    fn write_partition_sync(
        output_dir: &Path,
        partition_id: PartitionId,
        ops: &[&CopyOp],
    ) -> SaveResult<()> {
        let path = output_dir.join(format!("tensor.data_{}", partition_id.as_usize()));
        let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();

        // Read all source ranges, then write everything in one positioned open.
        let mut writes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            let source = sync::File::open(&op.source_path).map_err(SaveError::from)?;
            let data = source
                .read_at(op.source_offset, op.size)
                .map_err(SaveError::from)?;
            writes.push((op.dest_offset, data.as_ref().to_vec()));
        }

        let write_slices: Vec<WriteSlice<'_>> = writes
            .iter()
            .map(|(offset, data)| WriteSlice::new(*offset, data))
            .collect();
        let output = sync::File::create(&path).map_err(SaveError::from)?;
        output.set_len(total_size).map_err(SaveError::from)?;
        output
            .write_slices_at(WriteSlices::new(&write_slices).map_err(SaveError::from)?)
            .map_err(SaveError::from)?;
        output.sync_all().map_err(SaveError::from)
    }

    #[cfg(target_os = "linux")]
    fn write_partition_uring(
        output_dir: &Path,
        partition_id: PartitionId,
        ops: &[&CopyOp],
    ) -> SaveResult<()> {
        let path = output_dir.join(format!("tensor.data_{}", partition_id.as_usize()));
        let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();

        let mut writes: Vec<(u64, Vec<u8>)> = Vec::with_capacity(ops.len());
        for op in ops {
            let source = fastio::uring::File::open(&op.source_path).map_err(SaveError::from)?;
            let data = source
                .read_at(op.source_offset, op.size)
                .map_err(SaveError::from)?;
            writes.push((op.dest_offset, data.as_ref().to_vec()));
        }

        let write_slices: Vec<WriteSlice<'_>> = writes
            .iter()
            .map(|(offset, data)| WriteSlice::new(*offset, data))
            .collect();
        let output = fastio::uring::File::create(&path).map_err(SaveError::from)?;
        output.set_len(total_size).map_err(SaveError::from)?;
        output
            .write_slices_at(WriteSlices::new(&write_slices).map_err(SaveError::from)?)
            .map_err(SaveError::from)?;
        output.sync_all().map_err(SaveError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::Backend;
    use crate::formats::serverlessllm::Checkpoint as ServerlessLLMCheckpoint;
    use crate::formats::traits::{Checkpoint, Model};

    use safetensors::serialize;
    use safetensors::tensor::TensorView as StTensorView;
    use tempfile::TempDir;

    fn write_source_file(path: &Path, tensors: Vec<(&str, StTensorView<'_>)>) {
        let bytes = serialize(tensors, None).expect("serialize source file");
        std::fs::write(path, bytes).unwrap();
    }

    fn make_small_model_dir(tmp: &TempDir) -> PathBuf {
        let src_file = tmp.path().join("model.safetensors");
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let view = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        write_source_file(&src_file, vec![("weight", view)]);
        tmp.path().to_path_buf()
    }

    #[test]
    fn convert_rejects_zero_partitions() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out");
        let err = SafeTensorsToServerlessLLM::new(tmp.path(), &out, 0).unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_rejects_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out");
        let converter = SafeTensorsToServerlessLLM::new(tmp.path(), &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_single_file_roundtrip() {
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
    fn convert_multi_file_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let src_file1 = src.join("model-00001-of-00002.safetensors");
        let src_file2 = src.join("model-00002-of-00002.safetensors");

        let data1 = vec![0u8; 16];
        let data2 = vec![1u8; 32];
        let view1 = StTensorView::new(safetensors::Dtype::F32, vec![4], &data1).unwrap();
        let view2 = StTensorView::new(safetensors::Dtype::F32, vec![8], &data2).unwrap();
        write_source_file(&src_file1, vec![("a", view1)]);
        write_source_file(&src_file2, vec![("b", view2)]);

        let out = tmp.path().join("out");
        SafeTensorsToServerlessLLM::new(src, &out, 4)
            .unwrap()
            .convert_sync()
            .expect("convert multi-file");

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
        let src_file = src.join("model.safetensors");
        let data = vec![0u8; 16];
        let view = StTensorView::new(safetensors::Dtype::F32, vec![4], &data).unwrap();
        write_source_file(&src_file, vec![("a", view)]);

        std::fs::write(
            src.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"other.safetensors"}}"#,
        )
        .unwrap();

        let out = tmp.path().join("out");
        let converter = SafeTensorsToServerlessLLM::new(src, &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();

        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn convert_single_file_roundtrip_loads_same_tensor_content() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src_roundtrip_single");
        std::fs::create_dir_all(&src).unwrap();
        let src_file = src.join("model.safetensors");

        let a = vec![1u8, 2, 3, 4];
        let b = vec![5u8, 6, 7, 8, 9, 10, 11, 12];
        let view_a = StTensorView::new(safetensors::Dtype::U8, vec![4], &a).unwrap();
        let view_b = StTensorView::new(safetensors::Dtype::F32, vec![2], &b).unwrap();
        write_source_file(&src_file, vec![("a", view_a), ("b", view_b)]);

        let out = tmp.path().join("out_roundtrip_single");
        SafeTensorsToServerlessLLM::new(&src, &out, 2)
            .unwrap()
            .convert_sync()
            .unwrap();

        let converted = ServerlessLLMCheckpoint::load(&out, Backend::Sync).unwrap();
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
    fn convert_multi_file_roundtrip_loads_same_tensor_content() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path().join("src_roundtrip_multi");
        std::fs::create_dir_all(&src).unwrap();
        let src_file1 = src.join("model-00001-of-00002.safetensors");
        let src_file2 = src.join("model-00002-of-00002.safetensors");

        let a = vec![1u8, 2, 3, 4];
        let b = vec![5u8, 6, 7, 8];
        let view_a = StTensorView::new(safetensors::Dtype::U8, vec![4], &a).unwrap();
        let view_b = StTensorView::new(safetensors::Dtype::U8, vec![4], &b).unwrap();
        write_source_file(&src_file1, vec![("a", view_a)]);
        write_source_file(&src_file2, vec![("b", view_b)]);

        let out = tmp.path().join("out_roundtrip_multi");
        SafeTensorsToServerlessLLM::new(&src, &out, 4)
            .unwrap()
            .convert_sync()
            .unwrap();

        let converted = ServerlessLLMCheckpoint::load(&out, Backend::Sync).unwrap();
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
    fn convert_async_single_file_roundtrip() {
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
            let err = SafeTensorsToServerlessLLM::new(tmp.path(), &out, 0).unwrap_err();
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
    fn convert_duplicate_tensor_name_rejected() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let src_file1 = src.join("model-00001-of-00002.safetensors");
        let src_file2 = src.join("model-00002-of-00002.safetensors");

        let data = vec![0u8; 8];
        let view1 = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        let view2 = StTensorView::new(safetensors::Dtype::F32, vec![2], &data).unwrap();
        write_source_file(&src_file1, vec![("same_name", view1)]);
        write_source_file(&src_file2, vec![("same_name", view2)]);

        let out = tmp.path().join("out_dup");
        let converter = SafeTensorsToServerlessLLM::new(src, &out, 2).unwrap();
        let err = converter.convert_sync().unwrap_err();
        assert!(matches!(err, SaveError::InvalidInput(_)));
    }

    #[test]
    fn partition_balancing_many_tensors() {
        let tmp = TempDir::new().unwrap();
        let src = tmp.path();
        let src_file = src.join("model.safetensors");

        let data_a = vec![0u8; 16];
        let data_b = vec![0u8; 32];
        let data_c = vec![0u8; 64];
        let data_d = vec![0u8; 128];
        let va = StTensorView::new(safetensors::Dtype::U8, vec![16], &data_a).unwrap();
        let vb = StTensorView::new(safetensors::Dtype::U8, vec![32], &data_b).unwrap();
        let vc = StTensorView::new(safetensors::Dtype::U8, vec![64], &data_c).unwrap();
        let vd = StTensorView::new(safetensors::Dtype::U8, vec![128], &data_d).unwrap();
        write_source_file(&src_file, vec![("a", va), ("b", vb), ("c", vc), ("d", vd)]);

        let out = tmp.path().join("out_balance");
        SafeTensorsToServerlessLLM::new(src, &out, 2)
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
    fn discover_source_files_sorts_paths() {
        let tmp = TempDir::new().unwrap();
        let z = tmp.path().join("z_shard.safetensors");
        let a = tmp.path().join("a_shard.safetensors");
        let data = vec![0u8; 4];
        let v = StTensorView::new(safetensors::Dtype::U8, vec![4], &data).unwrap();
        write_source_file(&z, vec![("z", v.clone())]);
        let v2 = StTensorView::new(safetensors::Dtype::U8, vec![4], &data).unwrap();
        write_source_file(&a, vec![("a", v2)]);

        let files = ConversionPlan::discover_source_files(tmp.path()).unwrap();
        assert!(files[0].file_name().unwrap() < files[1].file_name().unwrap());
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
