//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! This module converts a directory of `*.safetensors` shards into a single ServerlessLLM
//! artifact. Input shards are discovered lexicographically, tensor names are de-duplicated across
//! shards, and tensors are assigned to partitions using a deterministic largest-first greedy
//! algorithm that balances partition sizes by bytes.
//!
//! Architecture:
//! 1. Metadata scan: collect tensor metadata (name, size, shape, stride, dtype, source shard)
//! 2. Planning: assign tensors to partitions using largest-first greedy by bytes
//! 3. Streaming write: write partition data directly without full materialization

use crate::backends;
use crate::formats::error::{WriterError, WriterResult};
use crate::formats::safetensors::Model;
use crate::formats::serverlessllm::{write_index, write_index_sync, writer::TensorWriteEntry};
use crate::formats::traits::TensorView;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// Convert a directory of `SafeTensors` shards to `ServerlessLLM` format.
///
/// The directory must contain one or more `*.safetensors` files. If a safetensors index JSON is
/// present, it is validated against the discovered shard set. The conversion is deterministic:
/// shards are processed lexicographically and tensors are assigned using a largest-first greedy
/// algorithm that balances partition sizes by bytes.
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

    let input_dir = Path::new(input_dir);
    let output_dir = Path::new(output_dir);

    let shard_paths = discover_safetensors_shards(input_dir)?;
    validate_index_manifest(input_dir, &shard_paths)?;

    let metadata = collect_tensor_metadata_async(&shard_paths).await?;

    let plan = build_conversion_plan(metadata, partition_count)?;

    tokio::fs::create_dir_all(output_dir).await?;

    write_partitions_async(output_dir, &plan).await?;

    let index_path = output_dir.join("tensor_index.json");
    write_index(&index_path, &plan.index).await?;

    Ok(())
}

/// Convert a directory of `SafeTensors` shards to `ServerlessLLM` format synchronously.
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

    let input_dir = Path::new(input_dir);
    let output_dir = Path::new(output_dir);

    let shard_paths = discover_safetensors_shards(input_dir)?;
    validate_index_manifest(input_dir, &shard_paths)?;

    let metadata = collect_tensor_metadata_sync(&shard_paths)?;

    let plan = build_conversion_plan(metadata, partition_count)?;

    std::fs::create_dir_all(output_dir).map_err(WriterError::from)?;

    write_partitions_sync(output_dir, &plan)?;

    let index_path = output_dir.join("tensor_index.json");
    write_index_sync(&index_path, &plan.index)?;

    Ok(())
}

/// Metadata for a single tensor from the source model.
#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    shard_path: PathBuf,
    size: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    dtype: String,
}

/// A tensor planned for output with partition assignment and offset.
#[derive(Debug)]
#[allow(dead_code)]
struct PlannedTensor {
    name: String,
    shard_path: PathBuf,
    size: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    dtype: String,
    partition_id: usize,
    offset: u64,
}

/// Complete conversion plan with partition assignments and index.
struct ConversionPlan {
    tensors: Vec<PlannedTensor>,
    index: HashMap<String, TensorWriteEntry>,
}

/// Collect tensor metadata from all shards concurrently.
async fn collect_tensor_metadata_async(shard_paths: &[PathBuf]) -> WriterResult<Vec<TensorMeta>> {
    let load_futures: Vec<_> = shard_paths
        .iter()
        .map(|path| {
            let path = path.clone();
            async move {
                let shard = backends::async_backend()
                    .load(&path)
                    .await
                    .map_err(WriterError::from)?;
                Result::<_, WriterError>::Ok((path, shard))
            }
        })
        .collect();

    let results = futures::future::try_join_all(load_futures).await?;

    let mut all_metadata = Vec::new();
    let mut seen_names = BTreeSet::new();

    for (shard_path, shard) in results {
        let model = Model::from_bytes(shard)
            .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;

        for name in model.tensor_names() {
            let tensor_name = name.as_ref().to_owned();
            if !seen_names.insert(tensor_name.clone()) {
                return Err(WriterError::InvalidInput(format!(
                    "duplicate tensor name across shards: {tensor_name}"
                )));
            }

            let view = model
                .tensor(name.as_ref())
                .map_err(|e| WriterError::InvalidInput(e.to_string()))?;

            let size = view.data().len();
            let shape: Vec<usize> = view.shape().to_vec();
            let stride = calculate_contiguous_stride(&shape);
            let dtype = dtype_str_to_serverlessllm(view.dtype())?.to_owned();

            all_metadata.push(TensorMeta {
                name: tensor_name,
                shard_path: shard_path.clone(),
                size,
                shape,
                stride,
                dtype,
            });
        }
    }

    Ok(all_metadata)
}

/// Collect tensor metadata from all shards using rayon parallel processing.
fn collect_tensor_metadata_sync(shard_paths: &[PathBuf]) -> WriterResult<Vec<TensorMeta>> {
    let results: Vec<Result<Vec<TensorMeta>, WriterError>> = shard_paths
        .par_iter()
        .map(|path| {
            let shard = backends::sync_backend()
                .load(path)
                .map_err(WriterError::from)?;

            let model = Model::from_bytes(shard)
                .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;

            let mut metadata = Vec::new();
            for name in model.tensor_names() {
                let view = model
                    .tensor(name.as_ref())
                    .map_err(|e| WriterError::InvalidInput(e.to_string()))?;

                let size = view.data().len();
                let shape: Vec<usize> = view.shape().to_vec();
                let stride = calculate_contiguous_stride(&shape);
                let dtype = dtype_str_to_serverlessllm(view.dtype())?.to_owned();

                metadata.push(TensorMeta {
                    name: name.as_ref().to_owned(),
                    shard_path: path.clone(),
                    size,
                    shape,
                    stride,
                    dtype,
                });
            }

            Ok(metadata)
        })
        .collect();

    let mut all_metadata = Vec::new();
    let mut seen_names = BTreeSet::new();

    for result in results {
        for meta in result? {
            if !seen_names.insert(meta.name.clone()) {
                return Err(WriterError::InvalidInput(format!(
                    "duplicate tensor name across shards: {}",
                    meta.name
                )));
            }
            all_metadata.push(meta);
        }
    }

    Ok(all_metadata)
}

/// Build conversion plan using largest-first greedy partitioning.
fn build_conversion_plan(
    mut metadata: Vec<TensorMeta>,
    partition_count: usize,
) -> WriterResult<ConversionPlan> {
    if metadata.is_empty() {
        return Err(WriterError::InvalidInput(
            "no tensors found in input directory".to_owned(),
        ));
    }

    // Largest-first greedy: sort by descending size, then ascending name for determinism
    metadata.sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.name.cmp(&b.name)));

    let mut partition_sizes: Vec<u64> = vec![0; partition_count];
    let mut tensors = Vec::with_capacity(metadata.len());
    let mut index: HashMap<String, TensorWriteEntry> = HashMap::with_capacity(metadata.len());

    for meta in metadata {
        // Find partition with smallest current size
        let smallest_partition = partition_sizes
            .iter()
            .enumerate()
            .min_by_key(|&(_, &size)| size)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let offset = partition_sizes[smallest_partition];
        partition_sizes[smallest_partition] += meta.size as u64;

        let size_u64 = meta.size as u64;

        let entry = TensorWriteEntry {
            offset,
            size: size_u64,
            shape: meta.shape.clone(),
            stride: meta.stride.clone(),
            dtype: meta.dtype.clone(),
            partition_id: smallest_partition,
        };

        index.insert(meta.name.clone(), entry);

        tensors.push(PlannedTensor {
            name: meta.name,
            shard_path: meta.shard_path,
            size: meta.size,
            shape: meta.shape,
            stride: meta.stride,
            dtype: meta.dtype,
            partition_id: smallest_partition,
            offset,
        });
    }

    Ok(ConversionPlan { tensors, index })
}

/// Write partition files concurrently.
async fn write_partitions_async(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    // Group tensors by partition
    let mut partition_tensors: HashMap<usize, Vec<&PlannedTensor>> = HashMap::new();
    for tensor in &plan.tensors {
        partition_tensors
            .entry(tensor.partition_id)
            .or_default()
            .push(tensor);
    }

    // Write each partition concurrently
    let write_futures: Vec<_> = partition_tensors
        .into_iter()
        .map(|(partition_id, tensors)| {
            let output_dir = output_dir.to_path_buf();
            async move { write_single_partition_async(&output_dir, partition_id, &tensors).await }
        })
        .collect();

    futures::future::try_join_all(write_futures).await?;

    Ok(())
}

async fn write_single_partition_async(
    output_dir: &Path,
    partition_id: usize,
    tensors: &[&PlannedTensor],
) -> WriterResult<()> {
    if tensors.is_empty() {
        let path = output_dir.join(format!("tensor.data_{}", partition_id));
        backends::async_backend()
            .write_all(&path, Vec::new())
            .await
            .map_err(WriterError::from)?;
        return Ok(());
    }

    // Group tensors by shard to minimize shard reopenings
    let mut tensors_by_shard: HashMap<&PathBuf, Vec<&PlannedTensor>> = HashMap::new();
    for tensor in tensors {
        tensors_by_shard
            .entry(&tensor.shard_path)
            .or_default()
            .push(tensor);
    }

    // Load all needed shards
    let mut shard_models: HashMap<PathBuf, Model> = HashMap::new();
    for shard_path in tensors_by_shard.keys() {
        let data = backends::async_backend()
            .load(shard_path)
            .await
            .map_err(WriterError::from)?;
        let model = Model::from_bytes(data)
            .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;
        shard_models.insert(shard_path.to_path_buf(), model);
    }

    // Collect bytes in order
    let mut partition_data = Vec::new();
    for tensor in tensors {
        let model = shard_models.get(&tensor.shard_path).ok_or_else(|| {
            WriterError::InvalidInput(format!("missing shard: {:?}", tensor.shard_path))
        })?;

        let view = model
            .tensor(tensor.name.as_str())
            .map_err(|e| WriterError::InvalidInput(e.to_string()))?;

        partition_data.extend_from_slice(view.data());
    }

    let path = output_dir.join(format!("tensor.data_{}", partition_id));
    backends::async_backend()
        .write_all(&path, partition_data)
        .await
        .map_err(WriterError::from)?;

    Ok(())
}

/// Write partition files using rayon parallel processing.
fn write_partitions_sync(output_dir: &Path, plan: &ConversionPlan) -> WriterResult<()> {
    // Group tensors by partition
    let mut partition_tensors: HashMap<usize, Vec<&PlannedTensor>> = HashMap::new();
    for tensor in &plan.tensors {
        partition_tensors
            .entry(tensor.partition_id)
            .or_default()
            .push(tensor);
    }

    // Write partitions in parallel
    let results: Vec<Result<(), WriterError>> = partition_tensors
        .into_par_iter()
        .map(|(partition_id, tensors)| {
            write_single_partition_sync(output_dir, partition_id, &tensors)
        })
        .collect();

    for result in results {
        result?;
    }

    Ok(())
}

fn write_single_partition_sync(
    output_dir: &Path,
    partition_id: usize,
    tensors: &[&PlannedTensor],
) -> Result<(), WriterError> {
    if tensors.is_empty() {
        let path = output_dir.join(format!("tensor.data_{}", partition_id));
        std::fs::write(&path, Vec::new())?;
        return Ok(());
    }

    // Group tensors by shard to minimize shard reopenings
    let mut tensors_by_shard: HashMap<&PathBuf, Vec<&PlannedTensor>> = HashMap::new();
    for tensor in tensors {
        tensors_by_shard
            .entry(&tensor.shard_path)
            .or_default()
            .push(tensor);
    }

    // Load all needed shards
    let mut shard_models: HashMap<PathBuf, Model> = HashMap::new();
    for shard_path in tensors_by_shard.keys() {
        let data = backends::sync_backend()
            .load(shard_path)
            .map_err(WriterError::from)?;
        let model = Model::from_bytes(data)
            .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;
        shard_models.insert(shard_path.to_path_buf(), model);
    }

    // Collect bytes in order
    let mut partition_data = Vec::new();
    for tensor in tensors {
        let model = shard_models.get(&tensor.shard_path).ok_or_else(|| {
            WriterError::InvalidInput(format!("missing shard: {:?}", tensor.shard_path))
        })?;

        let view = model
            .tensor(tensor.name.as_str())
            .map_err(|e| WriterError::InvalidInput(e.to_string()))?;

        partition_data.extend_from_slice(view.data());
    }

    let path = output_dir.join(format!("tensor.data_{}", partition_id));
    std::fs::write(&path, partition_data)?;

    Ok(())
}

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

fn dtype_str_to_serverlessllm(dtype: &str) -> WriterResult<&'static str> {
    let mapped = match dtype {
        "F32" => "torch.float32",
        "F16" => "torch.float16",
        "BF16" => "torch.bfloat16",
        "F64" => "torch.float64",
        "I32" => "torch.int32",
        "I16" => "torch.int16",
        "I8" => "torch.int8",
        "I64" => "torch.int64",
        "U32" => "torch.uint32",
        "U16" => "torch.uint16",
        "U8" => "torch.uint8",
        "U64" => "torch.uint64",
        "BOOL" => "torch.bool",
        _ => {
            return Err(WriterError::InvalidInput(format!(
                "unsupported dtype: {dtype}",
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
