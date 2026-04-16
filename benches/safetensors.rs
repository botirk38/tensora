//! SafeTensors benchmarks: tensora modes plus native baselines.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::{Path, PathBuf};
use tensora::formats::safetensors;

fn model_dirs() -> (String, PathBuf) {
    let id = std::env::var("TENSOR_STORE_MODEL_ID").unwrap_or_else(|_| {
        eprintln!("Set TENSOR_STORE_MODEL_ID to a Hugging Face model id (e.g. openai-community/gpt2).");
        std::process::exit(1);
    });
    let dir = tensora::hf_model::ensure_safetensors_hub_dir(&id).unwrap_or_else(|e| {
        eprintln!("Could not resolve model {id}: {e}");
        std::process::exit(1);
    });
    (id, dir)
}

fn collect_shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_file() {
                continue;
            }

            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.ends_with(".safetensors") {
                files.push(path);
            }
        }
    }
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    files
}

fn bench_default(c: &mut Criterion) {
    let (model_name, dir) = model_dirs();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("safetensors_default");
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.to_async(&rt).iter(|| async {
            let data = safetensors::Model::load(black_box(p)).await.unwrap();
            black_box((data.len(), data.tensor_names().len()))
        });
    });
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let (model_name, dir) = model_dirs();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("safetensors_sync");
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.iter(|| {
            let data = safetensors::Model::load_sync(black_box(p)).unwrap();
            black_box((data.len(), data.tensor_names().len()))
        });
    });
    group.finish();
}

fn bench_async(c: &mut Criterion) {
    let (model_name, dir) = model_dirs();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("safetensors_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.to_async(&rt).iter(|| async {
            let data = safetensors::Model::load_async(black_box(p)).await.unwrap();
            black_box((data.len(), data.tensor_names().len()))
        });
    });
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let (model_name, dir) = model_dirs();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("safetensors_mmap");
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.iter(|| {
            let data = safetensors::MmapModel::open(black_box(p)).unwrap();
            black_box((data.len(), data.tensor_names().len()))
        });
    });
    group.finish();
}

fn bench_native(c: &mut Criterion) {
    let (model_name, dir) = model_dirs();
    let mut group = c.benchmark_group("native_safetensors");
    for file in collect_shard_files(&dir) {
        let file_name = file.file_name().unwrap().to_string_lossy().to_string();
        let path_str = file.to_str().unwrap().to_string();
        group.bench_with_input(
            BenchmarkId::new(&model_name, &file_name),
            &path_str,
            |b, p| {
                b.iter(|| {
                    let bytes = std::fs::read(black_box(p)).unwrap();
                    let tensors = ::safetensors::SafeTensors::deserialize(&bytes).unwrap();
                    black_box((tensors.len(), bytes.len()))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_default,
    bench_sync,
    bench_async,
    bench_mmap,
    bench_native
);
criterion_main!(benches);
