use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use tensor_store::readers::safetensors;

fn load_safetensors_sync(path: &str) -> std::io::Result<(Vec<u8>, usize)> {
    let data = tensor_store::readers::safetensors::load_sync(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    let tensor_count = data.names().len();
    Ok((data.into_bytes(), tensor_count))
}

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

    // Parallel io_uring loading (8 chunks)
    c.bench_function("io_uring_safetensors_parallel_8_chunks", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let data = safetensors::load_parallel(black_box(test_file), 8)
                    .await
                    .unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            })
        });
    });

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

    c.bench_function("tokio_safetensors_parallel_8_chunks", |b| {
        b.to_async(&rt).iter(|| async {
            let data = safetensors::load_parallel(black_box(test_file), 8)
                .await
                .unwrap();
            let tensor_count = data.names().len();
            black_box((data, tensor_count))
        });
    });

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
            let result = load_safetensors_sync(black_box(test_file)).unwrap();
            black_box(result)
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(benches, bench_io_uring, bench_sync_safetensors);

#[cfg(not(target_os = "linux"))]
criterion_group!(benches, bench_tokio, bench_sync_safetensors);

criterion_main!(benches);
