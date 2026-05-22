//! Raw I/O backend micro-benchmarks.
//!
//! Isolates I/O layer performance from format parsing overhead by benchmarking
//! `SyncReader`, `AsyncReader`, and `io_uring::Reader` directly on real shard files.

mod bench_util;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use tensora::backends;

// ---------------------------------------------------------------------------
// Full-file load (single shard)
// ---------------------------------------------------------------------------

fn bench_sync_reader_load(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();

    let mut group = c.benchmark_group("backend_sync_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::SyncReader::new();
            let data = reader.load(black_box(&shard)).unwrap();
            black_box(data.len())
        });
    });
    group.finish();
}

fn bench_async_reader_load(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();

    let mut group = c.benchmark_group("backend_async_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let path = shard.clone();
            async move {
                let mut reader = backends::AsyncReader::new();
                let data = reader.load(black_box(&path)).await.unwrap();
                black_box(data.len())
            }
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_io_uring_reader_load(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();

    let mut group = c.benchmark_group("backend_io_uring_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::io_uring::Reader::new();
            let data = reader.load(black_box(&shard)).unwrap();
            black_box(data.len())
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Batch load (all shards)
// ---------------------------------------------------------------------------

fn bench_sync_reader_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shards = bench_util::collect_shard_files(&dir);
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("backend_sync_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::SyncReader::new();
            let results = reader.load_batch(black_box(&shards)).unwrap();
            black_box(results.len())
        });
    });
    group.finish();
}

fn bench_async_reader_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shards = bench_util::collect_shard_files(&dir);
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("backend_async_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let paths = shards.clone();
            async move {
                let mut reader = backends::AsyncReader::new();
                let results = reader.load_batch(black_box(&paths)).await.unwrap();
                black_box(results.len())
            }
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_io_uring_reader_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shards = bench_util::collect_shard_files(&dir);
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("backend_io_uring_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::io_uring::Reader::new();
            let results = reader.load_batch(black_box(&shards)).unwrap();
            black_box(results.len())
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Range batch (simulates per-tensor reads from a single shard)
// ---------------------------------------------------------------------------

fn build_range_requests(shard: &PathBuf) -> Vec<backends::BatchRequest> {
    let bytes = std::fs::read(shard).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut requests = Vec::new();
    for name in tensors.names() {
        let info = tensors.tensor(name).unwrap();
        let data = info.data();
        let offset_in_file = data.as_ptr() as usize - bytes.as_ptr() as usize;
        requests.push((shard.clone(), offset_in_file as u64, data.len()));
    }
    requests
}

fn bench_sync_reader_range_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let requests = build_range_requests(&shard);
    let total_bytes: u64 = requests.iter().map(|(_, _, len)| *len as u64).sum();

    let mut group = c.benchmark_group("backend_sync_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::SyncReader::new();
            let results = reader.load_range_batch(black_box(&requests)).unwrap();
            black_box(results.len())
        });
    });
    group.finish();
}

fn bench_async_reader_range_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let requests = build_range_requests(&shard);
    let total_bytes: u64 = requests.iter().map(|(_, _, len)| *len as u64).sum();

    let mut group = c.benchmark_group("backend_async_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let reqs = requests.clone();
            async move {
                let mut reader = backends::AsyncReader::new();
                let results = reader.load_range_batch(black_box(&reqs)).await.unwrap();
                black_box(results.len())
            }
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_io_uring_reader_range_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let requests = build_range_requests(&shard);
    let total_bytes: u64 = requests.iter().map(|(_, _, len)| *len as u64).sum();

    let mut group = c.benchmark_group("backend_io_uring_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.iter(|| {
            let mut reader = backends::io_uring::Reader::new();
            let results = reader.load_range_batch(black_box(&requests)).unwrap();
            black_box(results.len())
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Mmap open + page touch
// ---------------------------------------------------------------------------

fn bench_mmap_open_touch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();
    let shard_str = shard.to_str().unwrap().to_string();

    let mut group = c.benchmark_group("backend_mmap_open_touch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("open_touch", &slug), |b| {
        b.iter(|| {
            let mmap = backends::mmap::map(black_box(&shard_str)).unwrap();
            let data = mmap.as_slice();
            let page_size = 4096;
            let mut sum = 0u8;
            let mut offset = 0;
            while offset < data.len() {
                sum = sum.wrapping_add(data[offset]);
                offset += page_size;
            }
            if !data.is_empty() {
                sum = sum.wrapping_add(data[data.len() - 1]);
            }
            black_box(sum)
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
    bench_sync_reader_load,
    bench_async_reader_load,
    bench_io_uring_reader_load,
    bench_sync_reader_batch,
    bench_async_reader_batch,
    bench_io_uring_reader_batch,
    bench_sync_reader_range_batch,
    bench_async_reader_range_batch,
    bench_io_uring_reader_range_batch,
    bench_mmap_open_touch,
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_sync_reader_load,
    bench_async_reader_load,
    bench_sync_reader_batch,
    bench_async_reader_batch,
    bench_sync_reader_range_batch,
    bench_async_reader_range_batch,
    bench_mmap_open_touch,
);

criterion_main!(benches);
