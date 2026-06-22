//! Self-contained I/O backend micro-benchmarks for profiling.
//!
//! Unlike storage.rs, this does NOT require a downloaded model. It creates
//! synthetic files of realistic sizes and benchmarks all backends on them.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

#[cfg(target_os = "linux")]
use tensora::io::io_uring::IoUring;
use tensora::io::mmap::Mmap;
use tensora::io::sync::Sync;
use tensora::io::tokio::Tokio;
use tensora::io::{AsyncIo, BlockingIo, ByteRange, FileRange, MmapIo, WriteSlice, WriteSlices};

const SIZES: &[(usize, &str)] = &[
    (4 * 1024, "4KiB"),
    (1024 * 1024, "1MiB"),
    (64 * 1024 * 1024, "64MiB"),
    (256 * 1024 * 1024, "256MiB"),
];

fn create_test_file(dir: &TempDir, name: &str, size: usize) -> PathBuf {
    let path = dir.path().join(name);
    let data: Vec<u8> = (0u8..=255).cycle().take(size).collect();
    std::fs::write(&path, &data).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Sequential read benchmarks (full file)
// ---------------------------------------------------------------------------

fn bench_read_file(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let mut group = c.benchmark_group("io_read_file");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    for &(size, label) in SIZES {
        let path = create_test_file(&dir, &format!("read_{label}.bin"), size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("sync", label), |b| {
            b.iter(|| {
                let r = Sync::new();
                black_box(r.read_file(black_box(&path)).unwrap().len())
            });
        });

        #[cfg(target_os = "linux")]
        group.bench_function(BenchmarkId::new("io_uring", label), |b| {
            b.iter(|| {
                let r = IoUring::new();
                black_box(r.read_file(black_box(&path)).unwrap().len())
            });
        });

        let rt = tokio::runtime::Runtime::new().unwrap();
        group.bench_function(BenchmarkId::new("tokio", label), |b| {
            b.to_async(&rt).iter(|| {
                let p = path.clone();
                async move {
                    let r = Tokio::new();
                    black_box(r.read_file(black_box(&p)).await.unwrap().len())
                }
            });
        });

        group.bench_function(BenchmarkId::new("mmap", label), |b| {
            b.iter(|| {
                let m = Mmap::new();
                let region = m.map_file(black_box(&path)).unwrap();
                // Touch all pages to measure fault cost
                let data = region.as_ref();
                let mut sum = 0u8;
                let mut off = 0;
                while off < data.len() {
                    sum = sum.wrapping_add(data[off]);
                    off += 4096;
                }
                black_box(sum)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Range read benchmarks
// ---------------------------------------------------------------------------

fn bench_read_range(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let size = 64 * 1024 * 1024;
    let path = create_test_file(&dir, "range_target.bin", size);

    let mut group = c.benchmark_group("io_read_range");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    let range_sizes: &[(usize, &str)] = &[
        (4 * 1024, "4KiB"),
        (1024 * 1024, "1MiB"),
        (16 * 1024 * 1024, "16MiB"),
    ];

    for &(rsize, label) in range_sizes {
        let range = ByteRange::from_offset_len(1024, rsize).unwrap();
        group.throughput(Throughput::Bytes(rsize as u64));

        group.bench_function(BenchmarkId::new("sync", label), |b| {
            b.iter(|| {
                let r = Sync::new();
                black_box(r.read_range(black_box(&path), range).unwrap().len())
            });
        });

        #[cfg(target_os = "linux")]
        group.bench_function(BenchmarkId::new("io_uring", label), |b| {
            b.iter(|| {
                let r = IoUring::new();
                black_box(r.read_range(black_box(&path), range).unwrap().len())
            });
        });

        let rt = tokio::runtime::Runtime::new().unwrap();
        group.bench_function(BenchmarkId::new("tokio", label), |b| {
            b.to_async(&rt).iter(|| {
                let p = path.clone();
                async move {
                    let r = Tokio::new();
                    black_box(r.read_range(black_box(&p), range).await.unwrap().len())
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Batch range read benchmarks (simulates per-tensor reads)
// ---------------------------------------------------------------------------

fn bench_read_ranges_batch(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let size = 256 * 1024 * 1024;
    let path = create_test_file(&dir, "batch_target.bin", size);

    // Simulate 64 tensor reads of varying sizes spread across the file
    let num_ranges = 64;
    let chunk_size = size / num_ranges;
    let requests: Vec<(u64, usize)> = (0..num_ranges)
        .map(|i| {
            let offset = (i * chunk_size) as u64;
            let len = chunk_size - 1024; // leave gaps
            (offset, len)
        })
        .collect();
    let total_bytes: u64 = requests.iter().map(|(_, len)| *len as u64).sum();

    let mut group = c.benchmark_group("io_read_ranges_batch");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(2));
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function("sync", |b| {
        b.iter(|| {
            let r = Sync::new();
            let ranges: Vec<FileRange<'_>> = requests
                .iter()
                .map(|(off, len)| {
                    FileRange::new(&path, ByteRange::from_offset_len(*off, *len).unwrap())
                })
                .collect();
            black_box(r.read_ranges(black_box(&ranges)).unwrap().len())
        });
    });

    #[cfg(target_os = "linux")]
    group.bench_function("io_uring", |b| {
        b.iter(|| {
            let r = IoUring::new();
            let ranges: Vec<FileRange<'_>> = requests
                .iter()
                .map(|(off, len)| {
                    FileRange::new(&path, ByteRange::from_offset_len(*off, *len).unwrap())
                })
                .collect();
            black_box(r.read_ranges(black_box(&ranges)).unwrap().len())
        });
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("tokio", |b| {
        b.to_async(&rt).iter(|| {
            let reqs = requests.clone();
            let p = path.clone();
            async move {
                let r = Tokio::new();
                let ranges: Vec<FileRange<'_>> = reqs
                    .iter()
                    .map(|(off, len)| {
                        FileRange::new(&p, ByteRange::from_offset_len(*off, *len).unwrap())
                    })
                    .collect();
                black_box(r.read_ranges(black_box(&ranges)).await.unwrap().len())
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Write benchmarks
// ---------------------------------------------------------------------------

fn bench_write_file(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let mut group = c.benchmark_group("io_write_file");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    let write_sizes: &[(usize, &str)] = &[
        (1024 * 1024, "1MiB"),
        (64 * 1024 * 1024, "64MiB"),
    ];

    for &(size, label) in write_sizes {
        let data: Vec<u8> = (0u8..=255).cycle().take(size).collect();
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_function(BenchmarkId::new("sync", label), |b| {
            let path = dir.path().join(format!("sync_write_{label}.bin"));
            b.iter(|| {
                let w = Sync::new();
                w.write_file(black_box(&path), black_box(&data)).unwrap();
            });
        });

        #[cfg(target_os = "linux")]
        group.bench_function(BenchmarkId::new("io_uring", label), |b| {
            let path = dir.path().join(format!("uring_write_{label}.bin"));
            b.iter(|| {
                let w = IoUring::new();
                w.write_file(black_box(&path), black_box(&data)).unwrap();
            });
        });

        let rt = tokio::runtime::Runtime::new().unwrap();
        group.bench_function(BenchmarkId::new("tokio", label), |b| {
            let path = dir.path().join(format!("tokio_write_{label}.bin"));
            b.to_async(&rt).iter(|| {
                let p = path.clone();
                let d = data.clone();
                async move {
                    let w = Tokio::new();
                    w.write_file(black_box(&p), black_box(&d)).await.unwrap();
                }
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Positioned write benchmarks (batch slices)
// ---------------------------------------------------------------------------

fn bench_write_slices(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let mut group = c.benchmark_group("io_write_slices");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    // 32 non-overlapping slices of 1MiB each (32MiB total)
    let slice_size = 1024 * 1024;
    let num_slices = 32;
    let total_size = slice_size * num_slices;
    let slice_data: Vec<u8> = (0u8..=255).cycle().take(slice_size).collect();
    let write_slices: Vec<WriteSlice<'_>> = (0..num_slices)
        .map(|i| WriteSlice::new((i * slice_size) as u64, &slice_data))
        .collect();
    let ws = WriteSlices::new(&write_slices).unwrap();

    group.throughput(Throughput::Bytes(total_size as u64));

    group.bench_function("sync", |b| {
        let path = dir.path().join("sync_slices.bin");
        // Pre-create
        std::fs::write(&path, vec![0u8; total_size]).unwrap();
        b.iter(|| {
            let w = Sync::new();
            w.write_slices(black_box(&path), ws).unwrap();
        });
    });

    #[cfg(target_os = "linux")]
    group.bench_function("io_uring", |b| {
        let path = dir.path().join("uring_slices.bin");
        std::fs::write(&path, vec![0u8; total_size]).unwrap();
        b.iter(|| {
            let w = IoUring::new();
            w.write_slices(black_box(&path), ws).unwrap();
        });
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function("tokio", |b| {
        let path = dir.path().join("tokio_slices.bin");
        std::fs::write(&path, vec![0u8; total_size]).unwrap();
        b.to_async(&rt).iter(|| {
            let p = path.clone();
            async move {
                let w = Tokio::new();
                w.write_slices(black_box(&p), ws).await.unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_read_file,
    bench_read_range,
    bench_read_ranges_batch,
    bench_write_file,
    bench_write_slices,
);
criterion_main!(benches);
