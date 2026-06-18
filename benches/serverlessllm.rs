//! ServerlessLLM benchmarks: all I/O backends, tensor access patterns, mmap page-touch.

mod bench_util;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tensora::formats::serverlessllm;

// ---------------------------------------------------------------------------
// Full-model load benchmarks (one per I/O backend)
// ---------------------------------------------------------------------------

fn bench_default(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::serverlessllm_total_bytes(&sllm_dir);

    let mut group = c.benchmark_group("serverlessllm_default");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| async {
            let model = serverlessllm::Model::load(black_box(&dir_str))
                .await
                .unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_sync(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::serverlessllm_total_bytes(&sllm_dir);

    let mut group = c.benchmark_group("serverlessllm_sync");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = serverlessllm::Model::load_sync(black_box(&dir_str)).unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_tokio(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::serverlessllm_total_bytes(&sllm_dir);

    let mut group = c.benchmark_group("serverlessllm_tokio");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| async {
            let model = serverlessllm::Model::load_async(black_box(&dir_str))
                .await
                .unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_io_uring(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::serverlessllm_total_bytes(&sllm_dir);

    let mut group = c.benchmark_group("serverlessllm_io_uring");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = serverlessllm::Model::load_io_uring(black_box(&dir_str)).unwrap();
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

fn bench_mmap(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();
    let total_bytes = bench_util::serverlessllm_total_bytes(&sllm_dir);

    let mut group = c.benchmark_group("serverlessllm_mmap");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let model = serverlessllm::MmapModel::open(black_box(&dir_str)).unwrap();
            let bytes: usize = model
                .tensor_names()
                .iter()
                .filter_map(|name| model.tensor(name))
                .map(|t| {
                    let data = t.data();
                    if !data.is_empty() {
                        black_box(data[0]);
                        black_box(data[data.len() - 1]);
                    }
                    data.len()
                })
                .sum();
            black_box((model.len(), bytes))
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Tensor access pattern benchmarks (pre-loaded model)
// ---------------------------------------------------------------------------

fn bench_tensor_sequential(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();

    let model = serverlessllm::Model::load_sync(&dir_str).unwrap();
    let total_bytes: u64 = (&model)
        .into_iter()
        .map(|(_, t)| t.data().len() as u64)
        .sum();

    let mut group = c.benchmark_group("serverlessllm_tensor_sequential");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("scan", &slug), |b| {
        b.iter(|| {
            let mut bytes = 0usize;
            for name in model.tensor_names() {
                let t = model.tensor(name.as_ref()).unwrap();
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
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();

    let model = serverlessllm::Model::load_sync(&dir_str).unwrap();
    let names: Vec<String> = model.tensor_names().iter().map(|n| n.to_string()).collect();
    if names.is_empty() {
        return;
    }

    let mut group = c.benchmark_group("serverlessllm_tensor_random");
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
// Mmap tensor access (lazy open, then access each tensor)
// ---------------------------------------------------------------------------

fn bench_mmap_tensor_sequential(c: &mut Criterion) {
    let (model_id, _st_dir, sllm_dir) = bench_util::resolve_serverlessllm_model();
    let slug = bench_util::model_slug(&model_id);
    let dir_str = sllm_dir.to_str().unwrap().to_string();

    let mmap_model = serverlessllm::MmapModel::open(&dir_str).unwrap();
    let total_bytes: u64 = mmap_model
        .tensor_names()
        .iter()
        .filter_map(|name| mmap_model.tensor(name))
        .map(|t| t.data().len() as u64)
        .sum();

    let mut group = c.benchmark_group("serverlessllm_mmap_tensor_sequential");
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("scan", &slug), |b| {
        b.iter(|| {
            let mut bytes = 0usize;
            for name in mmap_model.tensor_names() {
                let t = mmap_model.tensor(name).unwrap();
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
    bench_mmap_tensor_sequential,
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
    bench_mmap_tensor_sequential,
);

criterion_main!(benches);
