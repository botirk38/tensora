use criterion::{Criterion, criterion_group, criterion_main};
use safetensors::SafeTensors;
use std::hint::black_box;
use zeropool::BufferPool;

fn load_safetensors_sync(
    path: &str,
    pool: Option<&BufferPool>,
) -> std::io::Result<(Vec<u8>, usize)> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let data = tensor_store::loaders::tokio::BasicLoader::load(path, pool).await?;
        let tensors = SafeTensors::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let tensor_count = tensors.names().len();
        Ok((data, tensor_count))
    })
}

#[cfg(target_os = "linux")]
fn bench_io_uring(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    // Standard io_uring (no IOPOLL, no O_DIRECT)
    c.bench_function("io_uring_safetensors", |b| {
        b.iter(|| {
            tokio_uring::start(async {
                let data = tensor_store::load_safetensors(black_box(test_file), None)
                    .await
                    .unwrap();
                let tensors = SafeTensors::deserialize(&data).unwrap();
                let tensor_count = tensors.names().len();
                black_box((data, tensor_count))
            })
        });
    });

    c.bench_function("io_uring_safetensors_with_pool", |b| {
        let pool = BufferPool::new();
        b.iter(|| {
            tokio_uring::start(async {
                let data = tensor_store::load_safetensors(black_box(test_file), Some(&pool))
                    .await
                    .unwrap();
                let tensors = SafeTensors::deserialize(&data).unwrap();
                let tensor_count = tensors.names().len();
                // Return buffer to pool for reuse
                pool.put(data);
                black_box(tensor_count)
            })
        });
    });

    c.bench_function("io_uring_safetensors_with_pool_pinned", |b| {
        let pool = BufferPool::builder().pinned_memory(true).build();
        b.iter(|| {
            tokio_uring::start(async {
                let data = tensor_store::load_safetensors(black_box(test_file), Some(&pool))
                    .await
                    .unwrap();
                let tensors = SafeTensors::deserialize(&data).unwrap();
                let tensor_count = tensors.names().len();
                // Return buffer to pool for reuse
                pool.put(data);
                black_box(tensor_count)
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
            let data = tensor_store::load_safetensors(black_box(test_file), None)
                .await
                .unwrap();
            let tensors = SafeTensors::deserialize(&data).unwrap();
            let tensor_count = tensors.names().len();
            black_box((data, tensor_count))
        });
    });

    c.bench_function("tokio_safetensors_with_pool", |b| {
        let pool = BufferPool::new();
        b.to_async(&rt).iter(|| async {
            let data = tensor_store::load_safetensors(black_box(test_file), Some(&pool))
                .await
                .unwrap();
            let tensors = SafeTensors::deserialize(&data).unwrap();
            let tensor_count = tensors.names().len();
            // Return buffer to pool for reuse
            pool.put(data);
            black_box(tensor_count)
        });
    });

    c.bench_function("tokio_safetensors_with_pool_pinned", |b| {
        let pool = BufferPool::builder().pinned_memory(true).build();
        b.to_async(&rt).iter(|| async {
            let data = tensor_store::load_safetensors(black_box(test_file), Some(&pool))
                .await
                .unwrap();
            let tensors = SafeTensors::deserialize(&data).unwrap();
            let tensor_count = tensors.names().len();
            // Return buffer to pool for reuse
            pool.put(data);
            black_box(tensor_count)
        });
    });
}

fn bench_sync_safetensors(c: &mut Criterion) {
    let test_file = "test_model.safetensors";

    c.bench_function("sync_safetensors", |b| {
        b.iter(|| {
            let result = load_safetensors_sync(black_box(test_file), None).unwrap();
            black_box(result)
        });
    });

    c.bench_function("sync_safetensors_with_pool", |b| {
        let pool = BufferPool::new();
        b.iter(|| {
            let (data, tensor_count) =
                load_safetensors_sync(black_box(test_file), Some(&pool)).unwrap();
            // Return buffer to pool for reuse
            pool.put(data);
            black_box(tensor_count)
        });
    });
}

#[cfg(target_os = "linux")]
criterion_group!(benches, bench_io_uring, bench_sync_safetensors);

#[cfg(not(target_os = "linux"))]
criterion_group!(benches, bench_tokio, bench_sync_safetensors);

criterion_main!(benches);
