//! Raw storage engine micro-benchmarks.
//!
//! Isolates I/O layer performance from format parsing overhead by benchmarking
//! `SyncStorage`, `TokioStorage`, `IoUringStorage`, and `MmapStorage` directly
//! on real shard files.

mod bench_util;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
#[cfg(target_os = "linux")]
use tensora::storage::io_uring::IoUringStorage;
use tensora::storage::mmap::MmapStorage;
use tensora::storage::sync::SyncStorage;
use tensora::storage::tokio::TokioStorage;
use tensora::storage::{AsyncReadableStorage, ByteRange, FileRange, MappableStorage, ReadableStorage};

// ---------------------------------------------------------------------------
// Full-file load (single shard)
// ---------------------------------------------------------------------------

fn bench_sync_reader_load(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();

    let mut group = c.benchmark_group("storage_sync_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let reader = SyncStorage::new();
            let data = reader.read_file(black_box(shard.as_path())).unwrap();
            black_box(data.len())
        });
    });
    group.finish();
}

fn bench_tokio_storage_load(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let shard_bytes = shard.metadata().unwrap().len();

    let mut group = c.benchmark_group("storage_tokio_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let path = shard.clone();
            async move {
                let reader = TokioStorage::new();
                let data = reader.read_file(black_box(path.as_path())).await.unwrap();
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

    let mut group = c.benchmark_group("storage_io_uring_load");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("load", &slug), |b| {
        b.iter(|| {
            let reader = IoUringStorage::new();
            let data = reader.read_file(black_box(shard.as_path())).unwrap();
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

    let mut group = c.benchmark_group("storage_sync_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.iter(|| {
            let reader = SyncStorage::new();
            let total_len: usize = black_box(&shards)
                .iter()
                .map(|path| reader.read_file(path.as_path()).unwrap().len())
                .sum();
            black_box(total_len)
        });
    });
    group.finish();
}

fn bench_tokio_storage_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shards = bench_util::collect_shard_files(&dir);
    let total_bytes = bench_util::safetensors_total_bytes(&dir);

    let mut group = c.benchmark_group("storage_tokio_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let paths = shards.clone();
            async move {
                let reader = TokioStorage::new();
                let mut total_len = 0usize;
                for path in black_box(&paths) {
                    total_len += reader.read_file(path.as_path()).await.unwrap().len();
                }
                black_box(total_len)
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

    let mut group = c.benchmark_group("storage_io_uring_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("load_batch", &slug), |b| {
        b.iter(|| {
            let reader = IoUringStorage::new();
            let total_len: usize = black_box(&shards)
                .iter()
                .map(|path| reader.read_file(path.as_path()).unwrap().len())
                .sum();
            black_box(total_len)
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Range batch (simulates per-tensor reads from a single shard)
// ---------------------------------------------------------------------------

fn build_range_requests(shard: &PathBuf) -> Vec<(PathBuf, u64, usize)> {
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

    let mut group = c.benchmark_group("storage_sync_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.iter(|| {
            let reader = SyncStorage::new();
            let ranges: Vec<FileRange<'_>> = requests
                .iter()
                .map(|(path, offset, len)| {
                    FileRange::new(path, ByteRange::from_offset_len(*offset, *len).unwrap())
                })
                .collect();
            let results = reader.read_ranges(black_box(&ranges)).unwrap();
            black_box(results.len())
        });
    });
    group.finish();
}

fn bench_tokio_storage_range_batch(c: &mut Criterion) {
    let (model_id, dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let shard = bench_util::first_safetensors_shard(&dir);
    let requests = build_range_requests(&shard);
    let total_bytes: u64 = requests.iter().map(|(_, _, len)| *len as u64).sum();

    let mut group = c.benchmark_group("storage_tokio_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let reqs = requests.clone();
            async move {
                let reader = TokioStorage::new();
                let ranges: Vec<FileRange<'_>> = reqs
                    .iter()
                    .map(|(path, offset, len)| {
                        FileRange::new(path, ByteRange::from_offset_len(*offset, *len).unwrap())
                    })
                    .collect();
                let results = reader.read_ranges(black_box(&ranges)).await.unwrap();
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

    let mut group = c.benchmark_group("storage_io_uring_range_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("range_batch", &slug), |b| {
        b.iter(|| {
            let reader = IoUringStorage::new();
            let ranges: Vec<FileRange<'_>> = requests
                .iter()
                .map(|(path, offset, len)| {
                    FileRange::new(path, ByteRange::from_offset_len(*offset, *len).unwrap())
                })
                .collect();
            let results = reader.read_ranges(black_box(&ranges)).unwrap();
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

    let mut group = c.benchmark_group("storage_mmap_open_touch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(shard_bytes));
    group.bench_function(BenchmarkId::new("open_touch", &slug), |b| {
        b.iter(|| {
            let storage = MmapStorage::new();
            let mmap = storage.map_file(black_box(shard.as_path())).unwrap();
            let data = mmap.as_ref();
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
    bench_tokio_storage_load,
    bench_io_uring_reader_load,
    bench_sync_reader_batch,
    bench_tokio_storage_batch,
    bench_io_uring_reader_batch,
    bench_sync_reader_range_batch,
    bench_tokio_storage_range_batch,
    bench_io_uring_reader_range_batch,
    bench_mmap_open_touch,
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_sync_reader_load,
    bench_tokio_storage_load,
    bench_sync_reader_batch,
    bench_tokio_storage_batch,
    bench_sync_reader_range_batch,
    bench_tokio_storage_range_batch,
    bench_mmap_open_touch,
);

criterion_main!(benches);
