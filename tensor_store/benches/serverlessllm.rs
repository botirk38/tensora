//! ServerlessLLM Benchmark Suite
//!
//! This benchmark suite provides performance measurements for the public API:
//!
//! 1. `cold_start_realistic_e2e` - Measures full cold-start time with io_uring
//!    - Includes runtime creation/teardown overhead
//!    - Measures complete end-to-end model loading
//!    - Use this for realistic performance expectations on Linux
//!
//! 2. `tokio_serverlessllm_all_tensors` - Async loading with tokio
//!    - Measures asynchronous I/O performance
//!    - Use this for comparison with sync versions
//!
//! 3. `sync_serverlessllm_all_tensors` - Synchronous loading baseline
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
//! # Profile tokio async loading
//! cargo flamegraph --bench serverlessllm -- --bench tokio_serverlessllm_all_tensors
//!
//! # Profile sync loading
//! cargo flamegraph --bench serverlessllm -- --bench sync_serverlessllm_all_tensors
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
use tensor_store::readers::serverlessllm;
use tensor_store::readers::traits::TensorMetadata;

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = std::path::Path::new("fixtures");
    let mut fixtures = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    let model_dir = entry.path().join("model_serverlessllm");
                    if model_dir.exists() && model_dir.is_dir() {
                        let model_name = entry.file_name().to_string_lossy().to_string();
                        fixtures.push((model_name, model_dir));
                    }
                }
            }
        }
    }

    // Sort by name for consistent ordering
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn bench_tokio_serverlessllm_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, dir) in &fixtures {
        let dir_str = dir.to_str().unwrap();
        c.bench_function(&format!("tokio_serverlessllm_load_{}", model_name), |b| {
            b.iter(|| {
                #[cfg(target_os = "linux")]
                {
                    tokio_uring::start(async {
                        let model = serverlessllm::load(black_box(dir_str)).unwrap();
                        let tensor_count = model.len();

                        let mut total_bytes = 0;
                        for (_name, tensor) in &model {
                            total_bytes += tensor.data().len();
                        }

                        black_box((total_bytes, tensor_count))
                    })
                }
                #[cfg(not(target_os = "linux"))]
                {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                        let model = serverlessllm::load(black_box(dir_str)).unwrap();
                        let tensor_count = model.len();

                        let mut total_bytes = 0;
                        for (_name, tensor) in &model {
                            total_bytes += tensor.data().len();
                        }

                        black_box((total_bytes, tensor_count))
                    })
                }
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
                let model = serverlessllm::load(black_box(dir_str)).unwrap();
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
                let tensor_count = model.len();

                let mut total_bytes = 0;
                for name in model.tensor_names() {
                    let tensor = model.tensor(name).unwrap();
                    total_bytes += tensor.data().len();
                }

                black_box((total_bytes, tensor_count))
            });
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_tokio_serverlessllm_load,
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
