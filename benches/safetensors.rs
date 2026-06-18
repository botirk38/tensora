//! SafeTensors benchmarks: all I/O backends, tensor access patterns, and native baselines.

mod bench_util;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tensora::formats::safetensors::Checkpoint as SafeTensorsCheckpoint;
use tensora::formats::traits::Checkpoint as _;
use tensora::formats::traits::Model as _;
use tensora::formats::traits::Tensor;

// ---------------------------------------------------------------------------
// Full-model load benchmarks (one per I/O backend)
// ---------------------------------------------------------------------------

fn bench_default(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("safetensors_default");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| async {
            let model = SafeTensorsCheckpoint::aload(black_box(&dir_str), tensora::AsyncBackend::Tokio).await.unwrap();
            black_box(bench_util::touch_all_tensors(&model))
        });
    });
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("safetensors_sync");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = SafeTensorsCheckpoint::load(black_box(&dir_str), tensora::Backend::Sync).unwrap();
            black_box(bench_util::touch_all_tensors(&model))
        });
    });
    group.finish();
}

fn bench_tokio(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("safetensors_tokio");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| async {
            let model = SafeTensorsCheckpoint::aload(black_box(&dir_str), tensora::AsyncBackend::Tokio).await.unwrap();
            black_box(bench_util::touch_all_tensors(&model))
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_io_uring(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("safetensors_io_uring");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = SafeTensorsCheckpoint::load(black_box(&dir_str), tensora::Backend::IoUring).unwrap();
            black_box(bench_util::touch_all_tensors(&model))
        });
    });
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("safetensors_mmap");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = SafeTensorsCheckpoint::open(black_box(&dir_str)).unwrap();
            black_box(bench_util::touch_all_tensors(&model))
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor access pattern benchmarks (pre-loaded model)
// ---------------------------------------------------------------------------

fn bench_tensor_sequential(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();

    let model = SafeTensorsCheckpoint::load(&dir_str, tensora::Backend::Sync).unwrap();
    let names: Vec<String> = model.tensor_names().map(|n| n.to_string()).collect();
    let total_bytes: u64 = names
        .iter()
        .filter_map(|n| model.tensor(n))
        .map(|t| t.data().len() as u64)
        .sum();

    let mut group = c.benchmark_group("safetensors_tensor_sequential");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("scan", &slug), |b| {
        b.iter(|| {
            let mut bytes = 0usize;
            for name in &names {
                let t = model.tensor(name).unwrap();
                let data = t.data();
                bytes += data.len();
                black_box(data[0]);
            }
            black_box(bytes)
        });
    });
    group.finish();
}

fn bench_tensor_random(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = dir.to_str().unwrap().to_string();

    let model = SafeTensorsCheckpoint::load(&dir_str, tensora::Backend::Sync).unwrap();
    let names: Vec<String> = model.tensor_names().map(|n| n.to_string()).collect();
    if names.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("safetensors_tensor_random");
    group.throughput(Throughput::Elements(names.len() as u64));
    group.bench_function(BenchmarkId::new("lookup", &slug), |b| {
        b.iter(|| {
            let mut bytes = 0usize;
            for name in names.iter().rev() {
                let t = model.tensor(name).unwrap();
                let data = t.data();
                bytes += data.len();
                black_box(data[0]);
            }
            black_box(bytes)
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Native safetensors crate baseline (per-shard)
// ---------------------------------------------------------------------------

fn bench_native(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shards = bench_util::collect_shard_files(&dir);

    let mut group = c.benchmark_group("native_safetensors");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    for file in &shards {
        let file_name = file.file_name().unwrap().to_string_lossy().to_string();
        let path_str = file.to_str().unwrap().to_string();
        let file_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);
        group.throughput(Throughput::Bytes(file_bytes));
        group.bench_function(BenchmarkId::new(&slug, &file_name), |b| {
            b.iter(|| {
                let bytes = std::fs::read(black_box(&path_str)).unwrap();
                let tensors = ::safetensors::SafeTensors::deserialize(&bytes).unwrap();
                let mut total = 0usize;
                for name in tensors.names() {
                    let t = tensors.tensor(name).unwrap();
                    total += t.data().len();
                    black_box(t.data()[0]);
                }
                black_box((tensors.len(), total))
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
criterion_group!(
    benches,
    bench_default,
    bench_sync,
    bench_tokio,
    bench_io_uring,
    bench_mmap,
    bench_tensor_sequential,
    bench_tensor_random,
    bench_native
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_default,
    bench_sync,
    bench_tokio,
    bench_mmap,
    bench_tensor_sequential,
    bench_tensor_random,
    bench_native
);

criterion_main!(benches);
