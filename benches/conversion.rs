//! SafeTensors → ServerlessLLM conversion pipeline benchmarks.
//!
//! Measures the full conversion pipeline (metadata scan → planning → materialization → index write)
//! for each storage-engine variant. Uses a fresh temp directory per iteration to avoid stale artifacts.

mod bench_util;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;
use tempfile::TempDir;

fn bench_conversion_default(c: &mut Criterion) {
    let (model_id, st_dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let total_bytes = bench_util::safetensors_total_bytes(&st_dir);
    let input_dir = st_dir.to_str().unwrap().to_string();
    let sizing = tensora::PartitionSizing::default_target();
    let partition_count = sizing.recommended_count(total_bytes).as_usize().max(1);

    let mut group = c.benchmark_group("conversion_default");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let input = input_dir.clone();
            async move {
                let tmp = TempDir::new().unwrap();
                let out = tmp.path().to_str().unwrap().to_string();
                let converter = tensora::SafeTensorsToServerlessLLM::new(
                    std::path::Path::new(&input),
                    std::path::Path::new(&out),
                    partition_count,
                )
                .unwrap();
                converter.convert_async().await.unwrap();
                black_box(())
            }
        });
    });
    group.finish();
}

fn bench_conversion_sync(c: &mut Criterion) {
    let (model_id, st_dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let total_bytes = bench_util::safetensors_total_bytes(&st_dir);
    let input_dir = st_dir.to_str().unwrap().to_string();
    let sizing = tensora::PartitionSizing::default_target();
    let partition_count = sizing.recommended_count(total_bytes).as_usize().max(1);

    let mut group = c.benchmark_group("conversion_sync");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let out = tmp.path().to_str().unwrap().to_string();
            let converter = tensora::SafeTensorsToServerlessLLM::new(
                std::path::Path::new(&input_dir),
                std::path::Path::new(&out),
                partition_count,
            )
            .unwrap();
            converter.convert_sync().unwrap();
            black_box(())
        });
    });
    group.finish();
}

fn bench_conversion_tokio(c: &mut Criterion) {
    let (model_id, st_dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let total_bytes = bench_util::safetensors_total_bytes(&st_dir);
    let input_dir = st_dir.to_str().unwrap().to_string();
    let sizing = tensora::PartitionSizing::default_target();
    let partition_count = sizing.recommended_count(total_bytes).as_usize().max(1);

    let mut group = c.benchmark_group("conversion_tokio");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    let rt = tokio::runtime::Runtime::new().unwrap();
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.to_async(&rt).iter(|| {
            let input = input_dir.clone();
            async move {
                let tmp = TempDir::new().unwrap();
                let out = tmp.path().to_str().unwrap().to_string();
                let converter = tensora::SafeTensorsToServerlessLLM::new(
                    std::path::Path::new(&input),
                    std::path::Path::new(&out),
                    partition_count,
                )
                .unwrap();
                converter.convert_async().await.unwrap();
                black_box(())
            }
        });
    });
    group.finish();
}

#[cfg(target_os = "linux")]
fn bench_conversion_io_uring(c: &mut Criterion) {
    let (model_id, st_dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let total_bytes = bench_util::safetensors_total_bytes(&st_dir);
    let input_dir = st_dir.to_str().unwrap().to_string();
    let sizing = tensora::PartitionSizing::default_target();
    let partition_count = sizing.recommended_count(total_bytes).as_usize().max(1);

    let mut group = c.benchmark_group("conversion_io_uring");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let out = tmp.path().to_str().unwrap().to_string();
            let converter = tensora::SafeTensorsToServerlessLLM::new(
                std::path::Path::new(&input_dir),
                std::path::Path::new(&out),
                partition_count,
            )
            .unwrap()
            .with_engine(tensora::ConversionEnginePreference::IoUring);
            converter.convert_sync().unwrap();
            black_box(())
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
    bench_conversion_default,
    bench_conversion_sync,
    bench_conversion_tokio,
    bench_conversion_io_uring,
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_conversion_default,
    bench_conversion_sync,
    bench_conversion_tokio,
);

criterion_main!(benches);
