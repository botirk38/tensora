use ::safetensors::SafeTensors;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use tensor_store::safetensors;

/// Touch one byte per page to ensure mmap benches trigger page faults.
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
                let model_path = entry.path().join("model.safetensors");
                if model_path.exists() {
                    let model_name = entry.file_name().to_string_lossy().to_string();
                    fixtures.push((model_name, model_path));
                }
            }
        }
    }

    // Sort by name for consistent ordering
    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

#[cfg(target_os = "linux")]
fn bench_io_uring_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(&format!("io_uring_safetensors_load_{}", model_name), |b| {
            b.iter(|| {
                tokio_uring::start(async {
                    let data = safetensors::load(black_box(path_str)).await.unwrap();
                    let tensor_count = data.names().len();
                    black_box((data, tensor_count))
                })
            });
        });
    }
}

#[cfg(target_os = "linux")]
fn bench_io_uring_parallel_n_chunks(c: &mut Criterion) {
    let fixtures = discover_fixtures();
    let num_cores = num_cpus::get();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(
            &format!("io_uring_safetensors_parallel_{}_{}", num_cores, model_name),
            |b| {
                b.iter(|| {
                    tokio_uring::start(async {
                        let data = safetensors::load_parallel(black_box(path_str), num_cores)
                            .await
                            .unwrap();
                        let tensor_count = data.tensors().names().len();
                        black_box((data, tensor_count))
                    })
                });
            },
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn bench_tokio_load(c: &mut Criterion) {
    let fixtures = discover_fixtures();
    let rt = tokio::runtime::Runtime::new().unwrap();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(&format!("tokio_safetensors_load_{}", model_name), |b| {
            b.to_async(&rt).iter(|| async {
                let data = safetensors::load(black_box(path_str)).await.unwrap();
                let tensor_count = data.names().len();
                black_box((data, tensor_count))
            });
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn bench_tokio_parallel(c: &mut Criterion) {
    let fixtures = discover_fixtures();
    let rt = tokio::runtime::Runtime::new().unwrap();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(
            &format!("tokio_safetensors_parallel_4_{}", model_name),
            |b| {
                b.to_async(&rt).iter(|| async {
                    let data = safetensors::load_parallel(black_box(path_str), 4)
                        .await
                        .unwrap();
                    let tensor_count = data.names().len();
                    black_box((data, tensor_count))
                });
            },
        );
    }
}

fn bench_sync_safetensors(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(&format!("sync_safetensors_load_{}", model_name), |b| {
            b.iter(|| {
                let data = safetensors::load_sync(black_box(path_str)).unwrap();
                let tensor_count = data.names().len();
                black_box((data.into_bytes(), tensor_count))
            });
        });
    }
}

fn bench_mmap_safetensors(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(&format!("mmap_safetensors_load_{}", model_name), |b| {
            b.iter(|| {
                let data = safetensors::load_mmap(black_box(path_str)).unwrap();
                let tensors = data.tensors();
                let names = tensors.names();
                let tensor_count = names.len();

                let mut checksum = 0u8;
                for name in names {
                    let tensor = tensors.tensor(name).unwrap();
                    let bytes = tensor.data();
                    checksum ^= touch_pages(bytes);
                }

                black_box((data, tensor_count, checksum))
            });
        });
    }
}

fn bench_original_safetensors(c: &mut Criterion) {
    let fixtures = discover_fixtures();

    for (model_name, path) in &fixtures {
        let path_str = path.to_str().unwrap();
        c.bench_function(&format!("original_safetensors_load_{}", model_name), |b| {
            b.iter(|| {
                let bytes = std::fs::read(black_box(path_str)).unwrap();
                let data = SafeTensors::deserialize(&bytes).unwrap();
                let tensor_count = data.names().len();
                black_box((tensor_count,))
            });
        });
    }
}

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_io_uring_load,
    bench_io_uring_parallel_n_chunks,
    bench_sync_safetensors,
    bench_mmap_safetensors,
    bench_original_safetensors
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_tokio_load,
    bench_tokio_parallel,
    bench_sync_safetensors,
    bench_mmap_safetensors,
    bench_original_safetensors
);

criterion_main!(benches);
