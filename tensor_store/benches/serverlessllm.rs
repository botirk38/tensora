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
//! Benchmarks expect a `test_model_serverlessllm/` directory containing:
//! - `tensor_index.json` - ServerlessLLM index file
//! - `tensor.data_0`, `tensor.data_1`, ... - Partitioned tensor data files

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tensor_store::readers::serverlessllm;
use tensor_store::readers::traits::TensorMetadata;

fn bench_tokio_serverlessllm(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";

    // === TOKIO SERVERLESSLLM ALL TENSORS ===
    // Measures async loading performance
    c.bench_function("tokio_serverlessllm_all_tensors", |b| {
        b.iter(|| {
            #[cfg(target_os = "linux")]
            {
                tokio_uring::start(async {
                    let model = serverlessllm::load(black_box(serverlessllm_dir))
                        .await
                        .unwrap();
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
                    let model = serverlessllm::load(black_box(serverlessllm_dir))
                        .await
                        .unwrap();
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

    // === TOKIO SERVERLESSLLM MMAP ALL TENSORS ===
    // Measures async mmap loading performance
    c.bench_function("tokio_serverlessllm_mmap_all_tensors", |b| {
        b.iter(|| {
            #[cfg(target_os = "linux")]
            {
                tokio_uring::start(async {
                    let model = serverlessllm::load_mmap(black_box(serverlessllm_dir))
                        .await
                        .unwrap();
                    let tensor_count = model.len();

                    let mut total_bytes = 0;
                    for name in model.tensor_names() {
                        let tensor = model.tensor(name).unwrap();
                        total_bytes += tensor.data().len();
                    }

                    black_box((total_bytes, tensor_count))
                })
            }
            #[cfg(not(target_os = "linux"))]
            {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async {
                    let model = serverlessllm::load_mmap(black_box(serverlessllm_dir))
                        .await
                        .unwrap();
                    let tensor_count = model.len();

                    let mut total_bytes = 0;
                    for name in model.tensor_names() {
                        let tensor = model.tensor(name).unwrap();
                        total_bytes += tensor.data().len();
                    }

                    black_box((total_bytes, tensor_count))
                })
            }
        });
    });
}

fn bench_sync_serverlessllm(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";

    // ServerlessLLM sync loading (load all tensors)
    c.bench_function("sync_serverlessllm_all_tensors", |b| {
        b.iter(|| {
            let model = serverlessllm::load_sync(black_box(serverlessllm_dir)).unwrap();
            let tensor_count = model.len();

            let mut total_bytes = 0;
            for (_name, tensor) in &model {
                total_bytes += tensor.data().len();
            }

            black_box((total_bytes, tensor_count))
        });
    });
}

fn bench_mmap_serverlessllm(c: &mut Criterion) {
    let serverlessllm_dir = "test_model_serverlessllm";

    // ServerlessLLM mmap loading (lazy, load all tensors)
    c.bench_function("mmap_serverlessllm_all_tensors", |b| {
        b.iter(|| {
            let model = serverlessllm::load_mmap_sync(black_box(serverlessllm_dir)).unwrap();
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

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_tokio_serverlessllm,
    bench_sync_serverlessllm,
    bench_mmap_serverlessllm
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_tokio_serverlessllm,
    bench_sync_serverlessllm,
    bench_mmap_serverlessllm
);

criterion_main!(benches);
