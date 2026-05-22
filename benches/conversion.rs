//! SafeTensors → ServerlessLLM conversion pipeline benchmarks.
//!
//! Measures the full conversion pipeline (metadata scan → planning → materialization → index write)
//! for each I/O backend variant. Uses a fresh temp directory per iteration to avoid stale artifacts.

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
    let partition_count = tensora::recommended_partition_count(total_bytes).max(1);

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
                tensora::convert_safetensors_to_serverlessllm(
                    black_box(&input),
                    &out,
                    partition_count,
                )
                .await
                .unwrap();
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
    let partition_count = tensora::recommended_partition_count(total_bytes).max(1);

    let mut group = c.benchmark_group("conversion_sync");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let out = tmp.path().to_str().unwrap().to_string();
            tensora::convert_safetensors_to_serverlessllm_sync(
                black_box(&input_dir),
                &out,
                partition_count,
            )
            .unwrap();
            black_box(())
        });
    });
    group.finish();
}

fn bench_conversion_async(c: &mut Criterion) {
    let (model_id, st_dir) = bench_util::resolve_safetensors_model();
    let slug = bench_util::model_slug(&model_id);
    let total_bytes = bench_util::safetensors_total_bytes(&st_dir);
    let input_dir = st_dir.to_str().unwrap().to_string();
    let partition_count = tensora::recommended_partition_count(total_bytes).max(1);

    let mut group = c.benchmark_group("conversion_async");
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
                tensora::convert_safetensors_to_serverlessllm_async(
                    black_box(&input),
                    &out,
                    partition_count,
                )
                .await
                .unwrap();
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
    let partition_count = tensora::recommended_partition_count(total_bytes).max(1);

    let mut group = c.benchmark_group("conversion_io_uring");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(3));
    group.throughput(Throughput::Bytes(total_bytes));
    group.bench_function(BenchmarkId::new("convert", &slug), |b| {
        b.iter(|| {
            let tmp = TempDir::new().unwrap();
            let out = tmp.path().to_str().unwrap().to_string();
            tensora::convert_safetensors_to_serverlessllm_io_uring(
                black_box(&input_dir),
                &out,
                partition_count,
            )
            .unwrap();
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
    bench_conversion_async,
    bench_conversion_io_uring,
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    benches,
    bench_conversion_default,
    bench_conversion_sync,
    bench_conversion_async,
);

criterion_main!(benches);
