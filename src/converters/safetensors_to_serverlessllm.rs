//! `SafeTensors` to `ServerlessLLM` conversion.
//!
//! This module provides functionality to convert `SafeTensors` format
//! checkpoints to `ServerlessLLM` format with configurable partitioning.
//!
//! # Conversion Process
//!
//! 1. Parse `SafeTensors` metadata using readers
//! 2. Convert dtypes and extract tensor data
//! 3. Partition tensors across multiple files
//! 4. Write `ServerlessLLM` format using writers
//!
//! # Example Usage (Future)
//!
//! ```rust,ignore
//! use tensor_store::converters::safetensors_to_serverlessllm;
//!
//! // Convert with 8 partitions
//! convert_safetensors_to_serverlessllm(
//!     "model.safetensors",
//!     "output_dir/",
//!     8,
//! ).await?;
//! ```

use crate::safetensors::Dtype;
use crate::serverlessllm::ServerlessLlmWriter;
use crate::serverlessllm::TensorEntry;
use crate::types::error::{WriterError, WriterResult};
use std::collections::HashMap;
use std::path::Path;

/// Convert `SafeTensors` to `ServerlessLLM` format
#[inline]
pub async fn convert_safetensors_to_serverlessllm(
    input_path: &str,
    output_dir: &str,
    partition_count: usize,
) -> WriterResult<()> {
    if partition_count == 0 {
        return Err(WriterError::InvalidInput(
            "partition_count must be greater than zero".to_owned(),
        ));
    }

    // Calculate optimal chunk count for parallel loading
    // Use thread-per-core by default, but ensure chunks stay under 2GB (isize::MAX limitation)
    const MAX_CHUNK_SIZE: usize = 2_000_000_000; // ~2GB, safely under isize::MAX (2,147,483,647)
    let file_size =
        usize::try_from(tokio::fs::metadata(input_path).await?.len()).map_err(|_e| {
            WriterError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "file too large",
            ))
        })?;
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let min_chunks_for_size = file_size.div_ceil(MAX_CHUNK_SIZE);
    let chunks = num_cpus.max(min_chunks_for_size);

    // Use parallel loading for better I/O performance
    let owned = crate::safetensors::load_parallel(input_path, chunks)
        .await
        .map_err(|e| WriterError::Io(std::io::Error::other(e.to_string())))?;
    let tensors = owned.tensors();
    let names = tensors.names();

    let mut blobs = Vec::with_capacity(names.len());
    for name in names {
        let view = tensors.tensor(name).map_err(WriterError::SafeTensors)?;
        let data = view.data().to_vec();
        let shape: Vec<i64> = view
            .shape()
            .iter()
            .map(|&d| i64::try_from(d).unwrap_or(i64::MAX))
            .collect();
        let stride = calculate_contiguous_stride(&shape);
        let dtype = dtype_to_serverlessllm(view.dtype())?.to_owned();
        blobs.push(TensorBlob {
            name: name.to_owned(),
            data,
            shape,
            stride,
            dtype,
        });
    }

    // Largest-first round-robin for better balance.
    blobs.sort_by(|a, b| b.data.len().cmp(&a.data.len()));

    let mut partitions: Vec<Vec<u8>> = vec![Vec::new(); partition_count];
    let mut index: HashMap<String, TensorEntry> = HashMap::with_capacity(blobs.len());

    for (i, blob) in blobs.into_iter().enumerate() {
        let partition_id = i.checked_rem(partition_count).unwrap_or(0);
        let partition = partitions
            .get_mut(partition_id)
            .ok_or_else(|| WriterError::InvalidInput("partition index out of bounds".to_owned()))?;
        let offset = u64::try_from(partition.len()).unwrap_or(u64::MAX);
        let size = u64::try_from(blob.data.len()).unwrap_or(u64::MAX);

        partition.extend_from_slice(&blob.data);

        index.insert(
            blob.name,
            TensorEntry {
                offset,
                size,
                shape: blob.shape,
                stride: blob.stride,
                dtype: blob.dtype,
                partition_id,
            },
        );
    }

    let writer = ServerlessLlmWriter::new();

    let out_dir = Path::new(output_dir);
    tokio::fs::create_dir_all(out_dir).await?;

    let index_path = out_dir.join("tensor_index.json");
    writer.write_index(&index_path, &index).await?;

    // Parallelize partition writing using concurrent futures
    // Uses platform-appropriate backend (io_uring on Linux, async_io elsewhere)
    let write_futures: Vec<_> = partitions
        .into_iter()
        .enumerate()
        .map(|(id, data)| {
            let part_path = out_dir.join(format!("tensor.data_{id}"));
            async move { writer.write_partition(&part_path, data).await }
        })
        .collect();

    // Execute all writes concurrently
    futures::future::try_join_all(write_futures).await?;

    Ok(())
}

fn dtype_to_serverlessllm(dtype: Dtype) -> WriterResult<&'static str> {
    let mapped = match dtype {
        Dtype::F32 => "torch.float32",
        Dtype::F16 => "torch.float16",
        Dtype::BF16 => "torch.bfloat16",
        Dtype::F64 => "torch.float64",
        Dtype::I32 => "torch.int32",
        Dtype::I16 => "torch.int16",
        Dtype::I8 => "torch.int8",
        Dtype::I64 => "torch.int64",
        Dtype::U32 => "torch.uint32",
        Dtype::U16 => "torch.uint16",
        Dtype::U8 => "torch.uint8",
        Dtype::U64 => "torch.uint64",
        Dtype::BOOL => "torch.bool",
        Dtype::F4
        | Dtype::F6_E2M3
        | Dtype::F6_E3M2
        | Dtype::F8_E5M2
        | Dtype::F8_E4M3
        | Dtype::F8_E8M0
        | Dtype::C64
        | _ => {
            return Err(WriterError::InvalidInput(format!(
                "unsupported dtype: {dtype:?}",
            )));
        }
    };

    Ok(mapped)
}

fn calculate_contiguous_stride(shape: &[i64]) -> Vec<i64> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut stride = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        let next_i = i + 1;
        let next_stride = stride.get(next_i).copied().unwrap_or(1);
        let next_shape = shape.get(next_i).copied().unwrap_or(1);
        if let Some(s) = stride.get_mut(i) {
            *s = next_stride.checked_mul(next_shape).unwrap_or(i64::MAX);
        }
    }
    stride
}

#[derive(Debug)]
struct TensorBlob {
    name: String,
    data: Vec<u8>,
    shape: Vec<i64>,
    stride: Vec<i64>,
    dtype: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::{Dtype, TensorView};
    use crate::serverlessllm::parse_index_sync;
    use crate::types::traits::TensorMetadata;
    use safetensors::serialize;
    use std::fs;
    use tempfile::TempDir;

    // Helper to create a simple SafeTensors file
    fn create_test_safetensors(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
        let path = dir.join(name);

        // Create two tensors with different sizes
        let data1 = vec![1u8, 2, 3, 4]; // 4 bytes
        let data2 = vec![5u8, 6, 7, 8, 9, 10]; // 6 bytes

        let tensor1 = TensorView::new(Dtype::U8, vec![4], &data1).expect("create tensor1");
        let tensor2 = TensorView::new(Dtype::U8, vec![2, 3], &data2).expect("create tensor2");

        let bytes = serialize([("weight", tensor1), ("bias", tensor2)], None)
            .expect("serialize tensors");

        fs::write(&path, bytes).expect("write safetensors file");
        path
    }

    // -----------------------------------------------------------------------
    // Unit Tests - Pure Functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_dtype_to_serverlessllm_supported() {
        assert_eq!(dtype_to_serverlessllm(Dtype::F32).unwrap(), "torch.float32");
        assert_eq!(dtype_to_serverlessllm(Dtype::F16).unwrap(), "torch.float16");
        assert_eq!(dtype_to_serverlessllm(Dtype::BF16).unwrap(), "torch.bfloat16");
        assert_eq!(dtype_to_serverlessllm(Dtype::I32).unwrap(), "torch.int32");
        assert_eq!(dtype_to_serverlessllm(Dtype::I8).unwrap(), "torch.int8");
        assert_eq!(dtype_to_serverlessllm(Dtype::U8).unwrap(), "torch.uint8");
        assert_eq!(dtype_to_serverlessllm(Dtype::BOOL).unwrap(), "torch.bool");
    }

    #[test]
    fn test_dtype_to_serverlessllm_unsupported() {
        // F4 and other exotic dtypes should be rejected
        let result = dtype_to_serverlessllm(Dtype::F4);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), WriterError::InvalidInput(_)));
    }

    #[test]
    fn test_calculate_contiguous_stride_empty() {
        let stride = calculate_contiguous_stride(&[]);
        assert_eq!(stride, Vec::<i64>::new());
    }

    #[test]
    fn test_calculate_contiguous_stride_1d() {
        let stride = calculate_contiguous_stride(&[10]);
        assert_eq!(stride, vec![1]);
    }

    #[test]
    fn test_calculate_contiguous_stride_2d() {
        let stride = calculate_contiguous_stride(&[3, 4]);
        assert_eq!(stride, vec![4, 1]); // Row-major: [cols, 1]
    }

    #[test]
    fn test_calculate_contiguous_stride_3d() {
        let stride = calculate_contiguous_stride(&[2, 3, 4]);
        assert_eq!(stride, vec![12, 4, 1]); // [depth*height, height, 1]
    }

    #[test]
    fn test_calculate_contiguous_stride_large() {
        let stride = calculate_contiguous_stride(&[100, 200, 300]);
        assert_eq!(stride, vec![60000, 300, 1]);
    }

    // -----------------------------------------------------------------------
    // Integration Tests - Full Conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_convert_rejects_zero_partitions() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            let result = convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                0,
            )
            .await;

            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), WriterError::InvalidInput(_)));
        });
    }

    #[test]
    fn test_convert_single_partition() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await
            .expect("conversion failed");
        });

        // Verify output files exist
        assert!(output.join("tensor_index.json").exists());
        assert!(output.join("tensor.data_0").exists());

        // Verify index is valid
        let index = parse_index_sync(output.join("tensor_index.json")).expect("parse index");
        assert_eq!(index.len(), 2);
        assert!(index.contains("weight"));
        assert!(index.contains("bias"));

        // Both tensors should be in partition 0
        assert_eq!(index.get("weight").unwrap().partition_id, 0);
        assert_eq!(index.get("bias").unwrap().partition_id, 0);
    }

    #[test]
    fn test_convert_multiple_partitions() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                2,
            )
            .await
            .expect("conversion failed");
        });

        // Verify output files exist
        assert!(output.join("tensor_index.json").exists());
        assert!(output.join("tensor.data_0").exists());
        assert!(output.join("tensor.data_1").exists());

        // Verify index
        let index = parse_index_sync(output.join("tensor_index.json")).expect("parse index");
        assert_eq!(index.len(), 2);

        // Tensors should be distributed across partitions (round-robin)
        let partition_0 = index.get("bias").unwrap().partition_id; // bias is larger (6 bytes)
        let partition_1 = index.get("weight").unwrap().partition_id; // weight is smaller (4 bytes)
        assert_eq!(partition_0, 0);
        assert_eq!(partition_1, 1);
    }

    #[test]
    fn test_convert_preserves_tensor_data() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await
            .expect("conversion failed");
        });

        // Load and verify tensor data
        let index = parse_index_sync(output.join("tensor_index.json")).expect("parse index");
        let weight_entry = index.get("weight").expect("weight tensor");
        let bias_entry = index.get("bias").expect("bias tensor");

        // Read partition file
        let partition_data = fs::read(output.join("tensor.data_0")).expect("read partition");

        // Extract tensor data from partition
        let weight_data = &partition_data[
            weight_entry.offset as usize..(weight_entry.offset + weight_entry.size) as usize
        ];
        let bias_data = &partition_data[
            bias_entry.offset as usize..(bias_entry.offset + bias_entry.size) as usize
        ];

        // Verify data matches original
        assert_eq!(weight_data, &[1u8, 2, 3, 4]);
        assert_eq!(bias_data, &[5u8, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_convert_preserves_metadata() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await
            .expect("conversion failed");
        });

        let index = parse_index_sync(output.join("tensor_index.json")).expect("parse index");

        // Check weight metadata
        let weight = index.get("weight").unwrap();
        assert_eq!(weight.shape, vec![4]);
        assert_eq!(weight.stride, vec![1]);
        assert_eq!(weight.dtype, "torch.uint8");
        assert_eq!(weight.size, 4);

        // Check bias metadata
        let bias = index.get("bias").unwrap();
        assert_eq!(bias.shape, vec![2, 3]);
        assert_eq!(bias.stride, vec![3, 1]); // Row-major stride
        assert_eq!(bias.dtype, "torch.uint8");
        assert_eq!(bias.size, 6);
    }

    #[test]
    fn test_convert_creates_output_directory() {
        let dir = TempDir::new().unwrap();
        let input = create_test_safetensors(dir.path(), "input.safetensors");
        let output = dir.path().join("nested").join("output");

        // Output directory doesn't exist yet
        assert!(!output.exists());

        tokio_uring::start(async {
            convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await
            .expect("conversion failed");
        });

        // Should create the directory
        assert!(output.exists());
        assert!(output.join("tensor_index.json").exists());
    }

    #[test]
    fn test_convert_missing_input_file() {
        let dir = TempDir::new().unwrap();
        let input = dir.path().join("nonexistent.safetensors");
        let output = dir.path().join("output");

        tokio_uring::start(async {
            let result = convert_safetensors_to_serverlessllm(
                input.to_str().unwrap(),
                output.to_str().unwrap(),
                1,
            )
            .await;

            assert!(result.is_err());
        });
    }
}
