//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! Converts a directory of `*.safetensors` shards into a ServerlessLLM artifact
//! (partition files + tensor index). The pipeline is:
//! 1. Metadata scan: parse SafeTensors headers to collect tensor descriptors
//! 2. Planning: assign tensors to partitions with locality-aware balancing
//! 3. Materialization: copy tensor bytes directly from source ranges to destination offsets
//! 4. Index write: produce `tensor_index.json`
//!
//! Four converter variants are exposed, all using the same pipeline with different I/O executors:
//! - `convert_safetensors_to_serverlessllm` — default (adaptive backend choice)
//! - `convert_safetensors_to_serverlessllm_sync` — synchronous I/O
//! - `convert_safetensors_to_serverlessllm_async` — Tokio async I/O
//! - `convert_safetensors_to_serverlessllm_io_uring` — Linux io_uring I/O

use crate::backends;
use crate::formats::error::{WriterError, WriterResult};
use crate::formats::serverlessllm::serializer::{TensorWriteEntry, write_index, write_index_sync};
use safetensors::SafeTensors;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public API — four converter variants
// ---------------------------------------------------------------------------

/// Convert SafeTensors shards to ServerlessLLM format using adaptive backend choice.
#[inline]
pub async fn convert_safetensors_to_serverlessllm(
    input_dir: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }
    let plan = build_plan(input_dir, partition_count).await?;
    let backend = choose_conversion_backend(&plan.stats);
    materialize(output_dir, &plan, backend).await
}

/// Convert SafeTensors shards to ServerlessLLM format using synchronous I/O.
#[inline]
pub fn convert_safetensors_to_serverlessllm_sync(
    input_dir: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }
    let plan = build_plan_sync(input_dir, partition_count)?;
    materialize_sync(output_dir, &plan)
}

/// Convert SafeTensors shards to ServerlessLLM format using Tokio async I/O.
#[inline]
pub async fn convert_safetensors_to_serverlessllm_async(
    input_dir: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }
    let plan = build_plan(input_dir, partition_count).await?;
    materialize_async(output_dir, &plan).await
}

/// Convert SafeTensors shards to ServerlessLLM format using Linux io_uring I/O.
#[cfg(target_os = "linux")]
#[inline]
pub fn convert_safetensors_to_serverlessllm_io_uring(
    input_dir: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }
    let plan = build_plan_sync(input_dir, partition_count)?;
    materialize_io_uring(output_dir, &plan)
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Describes a single tensor in its source shard.
#[derive(Debug, Clone)]
pub struct TensorSource {
    pub name: String,
    pub shard_id: usize,
    pub shard_path: PathBuf,
    pub source_offset: u64,
    pub size: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub dtype: String,
}

/// A single copy operation from source range to destination range.
#[derive(Debug, Clone)]
pub struct CopyOp {
    pub shard_id: usize,
    pub shard_path: PathBuf,
    pub source_offset: u64,
    pub dest_partition: usize,
    pub dest_offset: u64,
    pub size: usize,
}

/// Statistics about the conversion plan, used for backend selection.
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

/// Complete conversion plan with copy operations and index entries.
#[derive(Debug)]
pub struct ConversionPlan {
    pub copy_ops: Vec<CopyOp>,
    pub index: HashMap<String, TensorWriteEntry>,
    pub stats: ConversionStats,
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversionBackend {
    Sync,
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
}

fn choose_conversion_backend(stats: &ConversionStats) -> ConversionBackend {
    #[cfg(target_os = "linux")]
    {
        if stats.partition_count <= 1 && stats.shard_count <= 2 {
            return ConversionBackend::Sync;
        }

        let sync_score = score_sync(stats);
        let async_score = score_async(stats);
        let io_uring_score = score_io_uring(stats);

        if io_uring_score < sync_score && io_uring_score < async_score {
            ConversionBackend::IoUring
        } else if async_score < sync_score {
            ConversionBackend::TokioAsync
        } else {
            ConversionBackend::Sync
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        if stats.partition_count <= 1 {
            return ConversionBackend::Sync;
        }
        let sync_score = score_sync(stats);
        let async_score = score_async(stats);
        if async_score < sync_score {
            ConversionBackend::TokioAsync
        } else {
            ConversionBackend::Sync
        }
    }
}

fn score_sync(stats: &ConversionStats) -> f64 {
    const BASE: f64 = 1_000_000.0;
    const PER_BYTE: f64 = 1.0 / 2_000_000_000.0;
    const PER_SHARD: f64 = 50_000_000.0;
    const PER_PARTITION: f64 = 50_000_000.0;
    const FANOUT_PENALTY: f64 = 200_000_000.0;

    BASE + stats.total_bytes as f64 * PER_BYTE
        + stats.shard_count as f64 * PER_SHARD
        + stats.partition_count as f64 * PER_PARTITION
        + stats.max_shards_per_partition as f64 * FANOUT_PENALTY
}

fn score_async(stats: &ConversionStats) -> f64 {
    const BASE: f64 = 2_000_000.0;
    const PARALLELISM: f64 = 4.0;
    const PER_BYTE: f64 = 1.0 / 2_500_000_000.0;
    const PER_PARTITION: f64 = 30_000_000.0;
    const PER_COPY_OP: f64 = 500_000.0;
    const SMALL_COPY_PENALTY: f64 = 100_000_000.0;

    let small_copy_penalty = if stats.mean_copy_size < 1_048_576.0 {
        SMALL_COPY_PENALTY
    } else {
        0.0
    };

    BASE + stats.total_bytes as f64 * PER_BYTE / PARALLELISM
        + stats.partition_count as f64 * PER_PARTITION
        + stats.copy_op_count as f64 * PER_COPY_OP
        + small_copy_penalty
}

#[cfg(target_os = "linux")]
fn score_io_uring(stats: &ConversionStats) -> f64 {
    const BASE: f64 = 3_000_000.0;
    const PER_BYTE: f64 = 1.0 / 3_000_000_000.0;
    const PER_PARTITION: f64 = 20_000_000.0;
    const PER_CHUNK: f64 = 2_000_000.0;
    const FRAGMENTATION_PENALTY: f64 = 150_000_000.0;

    let fragmentation = if stats.mean_copy_size < 524_288.0 {
        FRAGMENTATION_PENALTY
    } else {
        0.0
    };

    let chunk_count = stats.total_bytes.div_ceil(128 * 1024 * 1024) as usize;

    BASE + stats.total_bytes as f64 * PER_BYTE
        + stats.partition_count as f64 * PER_PARTITION
        + chunk_count as f64 * PER_CHUNK
        + fragmentation
}

// ---------------------------------------------------------------------------
// Pipeline — scan, plan, materialize
// ---------------------------------------------------------------------------

async fn build_plan(input_dir: &str, partition_count: usize) -> WriterResult<ConversionPlan> {
    let input_dir = Path::new(input_dir);
    let shard_paths = discover_safetensors_shards(input_dir)?;
    validate_index_manifest(input_dir, &shard_paths)?;
    let tensors = scan_shards_async(&shard_paths).await?;
    build_plan_from_tensors(tensors, partition_count)
}

fn build_plan_sync(input_dir: &str, partition_count: usize) -> WriterResult<ConversionPlan> {
    let input_dir = Path::new(input_dir);
    let shard_paths = discover_safetensors_shards(input_dir)?;
    validate_index_manifest(input_dir, &shard_paths)?;
    let tensors = scan_shards_sync(&shard_paths)?;
    build_plan_from_tensors(tensors, partition_count)
}

fn build_plan_from_tensors(
    mut tensors: Vec<TensorSource>,
    partition_count: usize,
) -> WriterResult<ConversionPlan> {
    if tensors.is_empty() {
        return Err(WriterError::InvalidInput(
            "no tensors found in input directory".to_owned(),
        ));
    }

    let shard_count = tensors
        .iter()
        .map(|t| t.shard_id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1);

    let mut shard_sizes: Vec<u64> = vec![0; shard_count];
    for t in &tensors {
        shard_sizes[t.shard_id] += t.size as u64;
    }

    tensors.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(&b.name)));

    let mut partition_sizes: Vec<u64> = vec![0; partition_count];
    let mut partition_shards: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); partition_count];
    let mut copy_ops: Vec<CopyOp> = Vec::with_capacity(tensors.len());
    let mut index: HashMap<String, TensorWriteEntry> = HashMap::with_capacity(tensors.len());

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
        index.insert(
            tensor.name.clone(),
            TensorWriteEntry {
                offset,
                size: size_u64,
                shape: tensor.shape.clone(),
                stride: tensor.stride.clone(),
                dtype: tensor.dtype.clone(),
                partition_id: best_partition,
            },
        );

        copy_ops.push(CopyOp {
            shard_id: tensor.shard_id,
            shard_path: tensor.shard_path,
            source_offset: tensor.source_offset,
            dest_partition: best_partition,
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

// ---------------------------------------------------------------------------
// Scan — metadata-only SafeTensors header parsing via mmap
// ---------------------------------------------------------------------------

async fn scan_shards_async(shard_paths: &[PathBuf]) -> WriterResult<Vec<TensorSource>> {
    scan_shards_mmap(shard_paths)
}

fn scan_shards_sync(shard_paths: &[PathBuf]) -> WriterResult<Vec<TensorSource>> {
    scan_shards_mmap(shard_paths)
}

fn scan_shards_mmap(shard_paths: &[PathBuf]) -> WriterResult<Vec<TensorSource>> {
    use memmap2::Mmap;
    use std::fs::File;

    let mut all_tensors = Vec::new();
    let mut seen_names = std::collections::BTreeSet::new();

    for (shard_id, shard_path) in shard_paths.iter().enumerate() {
        let file = File::open(shard_path).map_err(WriterError::from)?;
        let mmap = unsafe { Mmap::map(&file).map_err(WriterError::from)? };
        let model = SafeTensors::deserialize(&mmap)
            .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;

        for name in model.names() {
            if !seen_names.insert(name.to_owned()) {
                return Err(WriterError::InvalidInput(format!(
                    "duplicate tensor name across shards: {name}"
                )));
            }

            let tensor = model
                .tensor(name)
                .map_err(|e| WriterError::InvalidInput(e.to_string()))?;
            let view = tensor.data();
            let offset = view.as_ptr() as usize - mmap.as_ptr() as usize;

            all_tensors.push(TensorSource {
                name: name.to_owned(),
                shard_id,
                shard_path: shard_path.clone(),
                source_offset: offset as u64,
                size: view.len(),
                shape: tensor.shape().to_vec(),
                stride: calculate_contiguous_stride(tensor.shape()),
                dtype: dtype_str_to_serverlessllm(tensor.dtype())?.to_owned(),
            });
        }
    }

    Ok(all_tensors)
}

// ---------------------------------------------------------------------------
// Materialization — streaming copy from source ranges to destination offsets
// ---------------------------------------------------------------------------

async fn materialize(
    output_dir: &str,
    plan: &ConversionPlan,
    backend: ConversionBackend,
) -> WriterResult<()> {
    let output_dir = Path::new(output_dir);
    tokio::fs::create_dir_all(output_dir).await?;

    match backend {
        ConversionBackend::Sync => materialize_sync_inner(output_dir, plan).await?,
        ConversionBackend::TokioAsync => materialize_async_inner(output_dir, plan).await?,
        #[cfg(target_os = "linux")]
        ConversionBackend::IoUring => materialize_io_uring_inner(output_dir, plan)?,
    }

    let index_path = output_dir.join("tensor_index.json");
    write_index(&index_path, &plan.index).await?;

    Ok(())
}

async fn materialize_async(output_dir: &str, plan: &ConversionPlan) -> WriterResult<()> {
    let output_dir = Path::new(output_dir);
    tokio::fs::create_dir_all(output_dir).await?;
    materialize_async_inner(output_dir, plan).await?;
    let index_path = output_dir.join("tensor_index.json");
    write_index(&index_path, &plan.index).await?;
    Ok(())
}

fn materialize_sync(output_dir: &str, plan: &ConversionPlan) -> WriterResult<()> {
    let output_dir = Path::new(output_dir);
    std::fs::create_dir_all(output_dir)?;
    materialize_sync_inner_sync(output_dir, plan)?;
    let index_path = output_dir.join("tensor_index.json");
    write_index_sync(&index_path, &plan.index)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn materialize_io_uring(output_dir: &str, plan: &ConversionPlan) -> WriterResult<()> {
    let output_dir = Path::new(output_dir);
    std::fs::create_dir_all(output_dir)?;
    materialize_io_uring_inner(output_dir, plan)?;
    let index_path = output_dir.join("tensor_index.json");
    write_index_sync(&index_path, &plan.index)?;
    Ok(())
}

async fn materialize_async_inner(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    // Group copy ops by destination partition
    let mut by_partition: HashMap<usize, Vec<&CopyOp>> = HashMap::new();
    for op in &plan.copy_ops {
        by_partition.entry(op.dest_partition).or_default().push(op);
    }

    // Write each partition sequentially to bound memory
    for (&partition_id, ops) in &by_partition {
        write_partition_async(output_dir, partition_id, ops).await?;
    }

    Ok(())
}

async fn materialize_sync_inner(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    let mut by_partition: HashMap<usize, Vec<&CopyOp>> = HashMap::new();
    for op in &plan.copy_ops {
        by_partition.entry(op.dest_partition).or_default().push(op);
    }

    for (&partition_id, ops) in &by_partition {
        write_partition_sync_single(output_dir, partition_id, ops)?;
    }

    Ok(())
}

fn materialize_sync_inner_sync(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    let mut by_partition: HashMap<usize, Vec<&CopyOp>> = HashMap::new();
    for op in &plan.copy_ops {
        by_partition.entry(op.dest_partition).or_default().push(op);
    }

    for (&partition_id, ops) in &by_partition {
        write_partition_sync_single(output_dir, partition_id, ops)?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn materialize_io_uring_inner(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    let mut by_partition: HashMap<usize, Vec<&CopyOp>> = HashMap::new();
    for op in &plan.copy_ops {
        by_partition.entry(op.dest_partition).or_default().push(op);
    }

    for (&partition_id, ops) in &by_partition {
        write_partition_sync_single(output_dir, partition_id, ops)?;
    }

    Ok(())
}

async fn write_partition_async(
    output_dir: &Path,
    partition_id: usize,
    ops: &[&CopyOp],
) -> WriterResult<()> {
    let path = output_dir.join(format!("tensor.data_{}", partition_id));

    // Pre-size partition file
    let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();
    {
        let f = std::fs::File::create(&path)?;
        f.set_len(total_size)?;
    }

    let mut writer = backends::AsyncWriter::create(&path)
        .await
        .map_err(WriterError::from)?;

    // Group ops by shard to minimize reopens
    let mut by_shard: HashMap<&PathBuf, Vec<&&CopyOp>> = HashMap::new();
    for op in ops {
        by_shard.entry(&op.shard_path).or_default().push(op);
    }

    for (shard_path, shard_ops) in by_shard {
        let mut reader = backends::AsyncReader::new();
        for op in shard_ops {
            let data = reader
                .load_range(shard_path, op.source_offset, op.size)
                .await
                .map_err(WriterError::from)?;
            writer
                .write_at(op.dest_offset, data.as_ref())
                .await
                .map_err(WriterError::from)?;
        }
    }

    writer.sync_all().await.map_err(WriterError::from)?;
    Ok(())
}

fn write_partition_sync_single(
    output_dir: &Path,
    partition_id: usize,
    ops: &[&CopyOp],
) -> WriterResult<()> {
    let path = output_dir.join(format!("tensor.data_{}", partition_id));

    // Pre-size partition file
    let total_size: u64 = ops.iter().map(|op| op.size as u64).sum();
    let f = std::fs::File::create(&path)?;
    f.set_len(total_size)?;

    let mut writer = backends::SyncWriter::create(&path)?;
    let mut reader = backends::SyncReader::new();

    let mut by_shard: HashMap<&PathBuf, Vec<&&CopyOp>> = HashMap::new();
    for op in ops {
        by_shard.entry(&op.shard_path).or_default().push(op);
    }

    for (shard_path, shard_ops) in by_shard {
        for op in shard_ops {
            let data = reader
                .load_range(shard_path.clone(), op.source_offset, op.size)
                .map_err(WriterError::from)?;
            writer
                .write_at(op.dest_offset, data.as_ref())
                .map_err(WriterError::from)?;
        }
    }

    writer.sync_all().map_err(WriterError::from)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn discover_safetensors_shards(input_dir: &Path) -> WriterResult<Vec<PathBuf>> {
    if !input_dir.is_dir() {
        return Err(WriterError::InvalidInput(format!(
            "input path is not a directory: {}",
            input_dir.display()
        )));
    }

    let mut shards = Vec::new();
    for entry in std::fs::read_dir(input_dir).map_err(WriterError::from)? {
        let entry = entry.map_err(WriterError::from)?;
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
        return Err(WriterError::InvalidInput(format!(
            "no .safetensors files found in {}",
            input_dir.display()
        )));
    }

    Ok(shards)
}

fn validate_index_manifest(input_dir: &Path, shard_paths: &[PathBuf]) -> WriterResult<()> {
    let mut index_files = Vec::new();
    for entry in std::fs::read_dir(input_dir).map_err(WriterError::from)? {
        let entry = entry.map_err(WriterError::from)?;
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
        let bytes = std::fs::read(&index_path).map_err(WriterError::from)?;
        let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
            WriterError::InvalidInput(format!(
                "failed to parse index manifest {}: {e}",
                index_path.display()
            ))
        })?;
        let weight_map = json
            .get("weight_map")
            .and_then(|value| value.as_object())
            .ok_or_else(|| {
                WriterError::InvalidInput(format!(
                    "index manifest {} is missing weight_map",
                    index_path.display()
                ))
            })?;

        let referenced: BTreeSet<String> = weight_map
            .values()
            .filter_map(|value| value.as_str().map(|s| s.to_owned()))
            .collect();

        if referenced != shard_names {
            return Err(WriterError::InvalidInput(format!(
                "index manifest {} does not match discovered shard set",
                index_path.display()
            )));
        }
    }

    Ok(())
}

fn dtype_str_to_serverlessllm(dtype: safetensors::Dtype) -> WriterResult<&'static str> {
    let mapped = match dtype {
        safetensors::Dtype::F32 => "torch.float32",
        safetensors::Dtype::F16 => "torch.float16",
        safetensors::Dtype::BF16 => "torch.bfloat16",
        safetensors::Dtype::F64 => "torch.float64",
        safetensors::Dtype::I32 => "torch.int32",
        safetensors::Dtype::I16 => "torch.int16",
        safetensors::Dtype::I8 => "torch.int8",
        safetensors::Dtype::I64 => "torch.int64",
        safetensors::Dtype::U32 => "torch.uint32",
        safetensors::Dtype::U16 => "torch.uint16",
        safetensors::Dtype::U8 => "torch.uint8",
        safetensors::Dtype::U64 => "torch.uint64",
        safetensors::Dtype::BOOL => "torch.bool",
        _ => {
            return Err(WriterError::InvalidInput(format!(
                "unsupported dtype: {:?}",
                dtype
            )));
        }
    };

    Ok(mapped)
}

fn calculate_contiguous_stride(shape: &[usize]) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
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
        let err = convert_safetensors_to_serverlessllm_sync(
            tmp.path().to_str().unwrap(),
            out.to_str().unwrap(),
            0,
        )
        .unwrap_err();
        assert!(matches!(err, WriterError::InvalidInput(_)));
    }

    #[test]
    fn convert_rejects_empty_directory() {
        let tmp = TempDir::new().unwrap();
        let out = tmp.path().join("out");
        let err = convert_safetensors_to_serverlessllm_sync(
            tmp.path().to_str().unwrap(),
            out.to_str().unwrap(),
            2,
        )
        .unwrap_err();
        assert!(matches!(err, WriterError::InvalidInput(_)));
    }

    #[test]
    fn convert_single_shard_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let src = make_small_model_dir(&tmp);
        let out = tmp.path().join("out");

        convert_safetensors_to_serverlessllm_sync(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            2,
        )
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
        convert_safetensors_to_serverlessllm_sync(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            4,
        )
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
        let err = convert_safetensors_to_serverlessllm_sync(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            2,
        )
        .unwrap_err();

        assert!(matches!(err, WriterError::InvalidInput(_)));
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
        convert_safetensors_to_serverlessllm_sync(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            2,
        )
        .unwrap();

        let converted = crate::formats::serverlessllm::Model::load_sync(&out).unwrap();
        let a_tensor = converted.tensor("a").unwrap();
        let b_tensor = converted.tensor("b").unwrap();

        assert_eq!(a_tensor.shape(), &[4]);
        assert_eq!(a_tensor.dtype(), "torch.uint8");
        assert_eq!(a_tensor.data(), a.as_slice());

        assert_eq!(b_tensor.shape(), &[2]);
        assert_eq!(b_tensor.dtype(), "torch.float32");
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
        convert_safetensors_to_serverlessllm_sync(
            src.to_str().unwrap(),
            out.to_str().unwrap(),
            4,
        )
        .unwrap();

        let converted = crate::formats::serverlessllm::Model::load_sync(&out).unwrap();
        let a_tensor = converted.tensor("a").unwrap();
        let b_tensor = converted.tensor("b").unwrap();

        assert_eq!(a_tensor.shape(), &[4]);
        assert_eq!(a_tensor.dtype(), "torch.uint8");
        assert_eq!(a_tensor.data(), a.as_slice());

        assert_eq!(b_tensor.shape(), &[4]);
        assert_eq!(b_tensor.dtype(), "torch.uint8");
        assert_eq!(b_tensor.data(), b.as_slice());
    }

    #[test]
    fn calculate_contiguous_stride_basic() {
        assert_eq!(calculate_contiguous_stride(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(calculate_contiguous_stride(&[5]), vec![1]);
        assert_eq!(calculate_contiguous_stride(&[]), Vec::<usize>::new());
    }

    #[test]
    fn dtype_mapping_covers_common_types() {
        assert_eq!(
            dtype_str_to_serverlessllm(safetensors::Dtype::F32).unwrap(),
            "torch.float32"
        );
        assert_eq!(
            dtype_str_to_serverlessllm(safetensors::Dtype::F16).unwrap(),
            "torch.float16"
        );
        assert_eq!(
            dtype_str_to_serverlessllm(safetensors::Dtype::BF16).unwrap(),
            "torch.bfloat16"
        );
        assert_eq!(
            dtype_str_to_serverlessllm(safetensors::Dtype::I64).unwrap(),
            "torch.int64"
        );
        assert_eq!(
            dtype_str_to_serverlessllm(safetensors::Dtype::U8).unwrap(),
            "torch.uint8"
        );
    }
}
