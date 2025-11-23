use criterion::{Criterion, criterion_group, criterion_main};
use safetensors::SafeTensors;
use std::hint::black_box;
use tensor_store::readers::safetensors;

#[cfg(target_os = "linux")]
fn bench_io_uring(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    // Standard io_uring (no IOPOLL, no O_DIRECT)
    c.bench_function("io_uring_safetensors", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let data = safetensors::load(black_box(test_file)).await.unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            })
        });
    });

    // Parallel io_uring loading (4 chunks)
    c.bench_function("io_uring_safetensors_parallel", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let data = safetensors::load_parallel(black_box(test_file), 4)
                    .await
                    .unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            })
        });
    });

    // Parallel io_uring loading (n chunks where n = number of CPU cores)
    let num_cores = num_cpus::get();
    c.bench_function(
        &format!("io_uring_safetensors_parallel_{}_chunks", num_cores),
        |b| {
            b.iter(|| {
                tokio_uring::start(async {
                    let data = safetensors::load_parallel(black_box(test_file), num_cores)
                        .await
                        .unwrap();
                    let tensor_count = data.tensors().names().len();
                    black_box((data, tensor_count))
                })
            });
        },
    );

    // Prewarmed io_uring loading (pool caches populated)
    c.bench_function("io_uring_safetensors_prewarmed", |b| {
        // Prewarm the buffer pool once before all iterations
        tokio_uring::start(async {
            for _ in 0..2 {
                let _warmup = safetensors::load(test_file).await.unwrap();
            }
        });

        b.iter(|| {
            tokio_uring::start(async {
                let data = safetensors::load(black_box(test_file)).await.unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            })
        });
    });

    // Mmap-backed loading (lazy)
    c.bench_function("io_uring_safetensors_mmap", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let data = safetensors::load_mmap(black_box(test_file)).await.unwrap();
                let tensor_count = data.tensors().names().len();
                black_box((data, tensor_count))
            })
        });
    });
}

#[cfg(not(target_os = "linux"))]
fn bench_tokio(c: &mut Criterion) {
    let test_file = "test_model.safetensors";
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("tokio_safetensors", |b| {
        b.to_async(&rt).iter(|| async {
            let data = safetensors::load(black_box(test_file)).await.unwrap();
            let tensor_count = data.names().len();
            black_box((data, tensor_count))
        });
    });

    c.bench_function("tokio_safetensors_parallel", |b| {
        b.to_async(&rt).iter(|| async {
            let data = safetensors::load_parallel(black_box(test_file), 4)
                .await
                .unwrap();
            let tensor_count = data.names().len();
            black_box((data, tensor_count))
        });
    });

    let num_cores = num_cpus::get();
    c.bench_function(
        &format!("tokio_safetensors_parallel_{}_chunks", num_cores),
        |b| {
            b.to_async(&rt).iter(|| async {
                let data = safetensors::load_parallel(black_box(test_file), num_cores)
                    .await
                    .unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            });
        },
    );

    c.bench_function("tokio_safetensors_prewarmed", |b| {
        // Prewarm the buffer pool once before all iterations
        rt.block_on(async {
            for _ in 0..2 {
                let _warmup = safetensors::load(test_file).await.unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            let data = safetensors::load(black_box(test_file)).await.unwrap();
            let tensor_count = data.names().len();
            black_box((data, tensor_count))
        });
    });
}

fn bench_sync_safetensors(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    c.bench_function("sync_safetensors", |b| {
        b.iter(|| {
            let data = safetensors::load_sync(black_box(test_file)).unwrap();
            let tensor_count = data.names().len();
            black_box((data.into_bytes(), tensor_count))
        });
    });
}

fn bench_mmap_safetensors(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    c.bench_function("mmap_safetensors", |b| {
        b.iter(|| {
            let data = safetensors::load_mmap_sync(black_box(test_file)).unwrap();
            let tensor_count = data.tensors().names().len();
            black_box((data, tensor_count))
        });
    });
}

fn bench_original_safetensors(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    c.bench_function("original_safetensors", |b| {
        b.iter(|| {
            let bytes = std::fs::read(black_box(test_file)).unwrap();
            let data = SafeTensors::deserialize(&bytes).unwrap();
            let tensor_count = data.names().len();
            black_box((tensor_count,))
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_io_uring,
    bench_sync_safetensors,
    bench_mmap_safetensors,
    bench_original_safetensors
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_tokio,
    bench_sync_safetensors,
    bench_mmap_safetensors,
    bench_original_safetensors
);

criterion_main!(benches);
