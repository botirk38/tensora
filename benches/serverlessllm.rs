//! ServerlessLLM Benchmark Suite
//!
//! This benchmark suite provides performance measurements for the public API:
//!
//! 1. `cold_start_realistic_e2e` - Measures full cold-start time with io_uring
//!    - Includes runtime creation/teardown overhead
//!    - Measures complete end-to-end model loading
//!    - Use this for realistic performance expectations on Linux
//!
//! 2. `io_uring_serverlessllm_load_{model}` - Async loading with io_uring (Linux only)
//!    - Measures asynchronous I/O performance with io_uring
//!    - Use this for high-performance async loading on Linux
//!
//! 3. `tokio_serverlessllm_load_{model}` - Async loading with tokio (non-Linux)
//!    - Measures asynchronous I/O performance with tokio
//!    - Use this for async loading on non-Linux platforms
//!
//! 4. `sync_serverlessllm_all_tensors` - Synchronous loading baseline
//!    - Measures synchronous I/O performance
//!    - Use this for comparison with async versions
//!
//! ## Profiling with Flamegraph
//!
//! To profile specific benchmarks with flamegraph:
//!
//! ```bash
//! # Profile cold start performance (Linux with io_uring)
//! cargo flamegraph --bench serverlessllm -- --bench cold_start_realistic_e2e
//!
//! # Profile io_uring async loading (Linux only)
//! cargo flamegraph --bench serverlessllm -- --bench io_uring_serverlessllm_load_{model}
//!
//! # Profile tokio async loading (non-Linux)
//! cargo flamegraph --bench serverlessllm -- --bench tokio_serverlessllm_load_{model}
//!
//! # Profile sync loading
//! cargo flamegraph --bench serverlessllm -- --bench sync_serverlessllm_load_{model}
//! ```
//!
//! ## Test Data
//!
//! Benchmarks expect a `model_serverlessllm/` directory containing:
//! - `tensor_index.json` - ServerlessLLM index file
//! - `tensor.data_0`, `tensor.data_1`, ... - Partitioned tensor data files

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use tensor_store::serverlessllm;
use tensor_store::types::traits::TensorMetadata;

/// Touch one byte per page so mmap benches pay the page fault cost.
fn touch_pages(data: &[u8]) -> u8 {
    const PAGE: usize = 4096;
    if data.is_empty() {
        return 0;
    }

    let mut idx = 0;
    let mut checksum = 0u8;
    while idx < data.len() {
        checksum ^= data[idx];
        idx += PAGE;
    }

    checksum ^ data[data.len() - 1]
}

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = std::path::Path::new("fixtures");
    let mut fixtures = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_dir()
            {
                let model_dir = entry.path().join("model_serverlessllm");
                if model_dir.exists() && model_dir.is_dir() {
                    let model_name = entry.file_name().to_string_lossy().to_string();
                    fixtures.push((model_name, model_dir));
                }
            }
        }
    }

    // Sort by name for consistent ordering
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

#[cfg(target_os = "linux")]
fn bench_io_uring_serverlessllm_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, dir) in &fixtures {
        let dir_str = dir.to_str().unwrap();
        c.bench_function(
            &format!("io_uring_serverlessllm_load_{}", model_name),
            |b| {
                b.iter(|| {
                    tokio_uring::start(async {
                        let model = serverlessllm::load(black_box(dir_str)).await.unwrap();
                        let tensor_count = model.len();

                        let mut total_bytes = 0;
                        for (_name, tensor) in &model {
                            total_bytes += tensor.data().len();
                        }

                        black_box((total_bytes, tensor_count))
                    })
                });
            },
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn bench_tokio_serverlessllm_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, dir) in &fixtures {
        let dir_str = dir.to_str().unwrap();
        c.bench_function(&format!("tokio_serverlessllm_load_{}", model_name), |b| {
            b.iter(|| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let model = serverlessllm::load(black_box(dir_str)).await.unwrap();
                    let tensor_count = model.len();

                    let mut total_bytes = 0;
                    for (_name, tensor) in &model {
                        total_bytes += tensor.data().len();
                    }

                    black_box((total_bytes, tensor_count))
                })
            });
        });
    }
}

fn bench_sync_serverlessllm_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, dir) in &fixtures {
        let dir_str = dir.to_str().unwrap();
        c.bench_function(&format!("sync_serverlessllm_load_{}", model_name), |b| {
            b.iter(|| {
                let model = serverlessllm::load_sync(black_box(dir_str)).unwrap();
                let tensor_count = model.len();

                let mut total_bytes = 0;
                for (_name, tensor) in &model {
                    total_bytes += tensor.data().len();
                }

                black_box((total_bytes, tensor_count))
            });
        });
    }
}

fn bench_mmap_serverlessllm_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, dir) in &fixtures {
        let dir_str = dir.to_str().unwrap();
        c.bench_function(&format!("mmap_serverlessllm_load_{}", model_name), |b| {
            b.iter(|| {
                let model = serverlessllm::load_mmap(black_box(dir_str)).unwrap();
                let tensor_names = model.tensor_names();
                let tensor_count = tensor_names.len();

                let mut total_bytes = 0;
                let mut checksum = 0u8;
                for name in tensor_names {
                    let tensor = model.tensor(name).unwrap();
                    let data = tensor.data();
                    total_bytes += data.len();
                    checksum ^= touch_pages(data);
                }

                black_box((total_bytes, tensor_count, checksum))
            });
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_io_uring_serverlessllm_load,
    bench_sync_serverlessllm_load,
    bench_mmap_serverlessllm_load
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_tokio_serverlessllm_load,
    bench_sync_serverlessllm_load,
    bench_mmap_serverlessllm_load
);

criterion_main!(benches);
