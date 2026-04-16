//! SafeTensors profiling scenarios.

use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensora::formats::safetensors;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};
use crate::stats::summarize;

fn resolved_dir(config: &ProfileConfig) -> Result<(String, PathBuf), ProfileError> {
    let dir = tensora::hf_model::ensure_safetensors_hub_dir(&config.model_id)
        .map_err(|e| ProfileError::new(e.to_string()))?;
    Ok((config.model_id.clone(), dir))
}

fn total_file_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".safetensors") {
            total += fs::metadata(path)?.len();
        }
    }
    Ok(total)
}

fn profile_load_sync(config: &ProfileConfig) -> ProfileResult {
    let (label, dir) = resolved_dir(config)?;
    let iterations = config.normalized_iterations();
    let total_bytes = total_file_bytes(&dir)?;
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running sync safetensors load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let data = safetensors::Model::load_sync(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            data.len(),
            total_bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((data.len(), total_bytes));
    }

    if let Some(summary) = summarize(&durations) {
        println!(
            "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
            summary.mean_ms, summary.min_ms, summary.max_ms
        );
    }
    Ok(())
}

fn profile_load_mmap(config: &ProfileConfig) -> ProfileResult {
    let (label, dir) = resolved_dir(config)?;
    let iterations = config.normalized_iterations();
    let total_bytes = total_file_bytes(&dir)?;
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running mmap safetensors load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let data = safetensors::MmapModel::open(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            data.len(),
            total_bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((data.len(), total_bytes));
    }

    if let Some(summary) = summarize(&durations) {
        println!(
            "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
            summary.mean_ms, summary.min_ms, summary.max_ms
        );
    }
    Ok(())
}

fn profile_load_async(config: &ProfileConfig, default: bool) -> ProfileResult {
    let (label, dir) = resolved_dir(config)?;
    let rt = tokio::runtime::Runtime::new()?;
    let iterations = config.normalized_iterations();
    let total_bytes = total_file_bytes(&dir)?;
    let mut durations = Vec::with_capacity(iterations);
    let case_label = if default { "default" } else { "async" };

    println!(
        "Running {case_label} safetensors load for '{}' ({iterations}x)",
        label
    );
    rt.block_on(async {
        for i in 0..iterations {
            let start = Instant::now();
            let data = if default {
                safetensors::Model::load(&dir).await?
            } else {
                safetensors::Model::load_async(&dir).await?
            };
            let elapsed = start.elapsed();
            durations.push(elapsed);
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                data.len(),
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((data.len(), total_bytes));
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
        Ok::<_, tensora::ReaderError>(())
    })?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn profile_load_io_uring(config: &ProfileConfig) -> ProfileResult {
    let (label, dir) = resolved_dir(config)?;
    let iterations = config.normalized_iterations();
    let total_bytes = total_file_bytes(&dir)?;
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running io-uring safetensors load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let data = safetensors::Model::load_io_uring(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            data.len(),
            total_bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((data.len(), total_bytes));
    }

    if let Some(summary) = summarize(&durations) {
        println!(
            "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
            summary.mean_ms, summary.min_ms, summary.max_ms
        );
    }
    Ok(())
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "default" => profile_load_async(config, true),
        "sync" => profile_load_sync(config),
        "async" => profile_load_async(config, false),
        "mmap" => profile_load_mmap(config),
        #[cfg(target_os = "linux")]
        "io-uring" => profile_load_io_uring(config),
        #[cfg(not(target_os = "linux"))]
        "io-uring" => Err(ProfileError::new("io-uring is only available on Linux").into()),
        other => Err(ProfileError::new(format!("Unknown safetensors case '{}'", other)).into()),
    }
}
