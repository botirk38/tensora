//! ServerlessLLM benchmarks: default, sync, async, mmap.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use tensora::formats::serverlessllm;

fn model_dir() -> (String, PathBuf) {
    let id = std::env::var("TENSOR_STORE_MODEL_ID").unwrap_or_else(|_| {
        eprintln!(
            "Set TENSOR_STORE_MODEL_ID to a Hugging Face model id (e.g. openai-community/gpt2)."
        );
        std::process::exit(1);
    });
    let safetensors_dir = tensora::hf_model::ensure_safetensors_hub_dir(&id).unwrap_or_else(|e| {
        eprintln!("Could not download safetensors for {id}: {e}");
        std::process::exit(1);
    });
    let dir = tensora::hf_model::ensure_serverlessllm_cache_dir(&id, &safetensors_dir)
        .unwrap_or_else(|e| {
            eprintln!("Could not build ServerlessLLM layout for {id}: {e}");
            std::process::exit(1);
        });
    (id, dir)
}

fn bench_default(c: &mut Criterion) {
    let (model_name, dir) = model_dir();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("serverlessllm_default");
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.to_async(&rt).iter(|| async {
            let model = serverlessllm::Model::load(black_box(p)).await.unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let (model_name, dir) = model_dir();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("serverlessllm_sync");
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.iter(|| {
            let model = serverlessllm::Model::load_sync(black_box(p)).unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_async(c: &mut Criterion) {
    let (model_name, dir) = model_dir();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("serverlessllm_async");
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.to_async(&rt).iter(|| async {
            let model = serverlessllm::Model::load_async(black_box(p))
                .await
                .unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let (model_name, dir) = model_dir();
    let dir_str = dir.to_str().unwrap().to_string();
    let mut group = c.benchmark_group("serverlessllm_mmap");
    group.bench_with_input(BenchmarkId::new("load", &model_name), &dir_str, |b, p| {
        b.iter(|| {
            let model = serverlessllm::MmapModel::open(black_box(p)).unwrap();
            let bytes: usize = model
                .tensor_names()
                .iter()
                .map(|name| model.tensor(name).unwrap().data().len())
                .sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

criterion_group!(benches, bench_default, bench_sync, bench_async, bench_mmap);
criterion_main!(benches);
