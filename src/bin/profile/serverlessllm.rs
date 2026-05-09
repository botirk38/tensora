//! ServerlessLLM profiling scenarios.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use tensora::formats::serverlessllm;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};
use crate::evict;
use crate::stats::summarize;

fn resolved_dir(config: &ProfileConfig) -> Result<(String, PathBuf), ProfileError> {
    let safetensors_dir = tensora::hf_model::ensure_safetensors_hub_dir(&config.model_id)
        .map_err(|e| ProfileError::new(e.to_string()))?;
    let dir = tensora::hf_model::ensure_serverlessllm_cache_dir(&config.model_id, &safetensors_dir)
        .map_err(|e| ProfileError::new(e.to_string()))?;
    Ok((config.model_id.clone(), dir))
}

fn profile_load_sync(config: &ProfileConfig) -> ProfileResult {
    let (label, dir) = resolved_dir(config)?;
    let iterations = config.normalized_iterations();
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running sync serverlessllm load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let model = serverlessllm::Model::load_sync(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            model.len(),
            bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((model.len(), bytes));
        if config.evict_page_cache {
            evict::evict_page_cache(&dir);
        }
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
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running mmap serverlessllm load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let model = serverlessllm::MmapModel::open(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        let bytes: usize = model
            .tensor_names()
            .iter()
            .map(|name| model.tensor(name).unwrap().data().len())
            .sum();
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            model.len(),
            bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((model.len(), bytes));
        if config.evict_page_cache {
            evict::evict_page_cache(&dir);
        }
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
    let mut durations = Vec::with_capacity(iterations);
    let case_label = if default { "default" } else { "async" };

    println!(
        "Running {case_label} serverlessllm load for '{}' ({iterations}x)",
        label
    );
    rt.block_on(async {
        for i in 0..iterations {
            let start = Instant::now();
            let model = if default {
                serverlessllm::Model::load(&dir).await?
            } else {
                serverlessllm::Model::load_async(&dir).await?
            };
            let elapsed = start.elapsed();
            durations.push(elapsed);
            let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                model.len(),
                bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((model.len(), bytes));
            if config.evict_page_cache {
                evict::evict_page_cache(&dir);
            }
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
    let mut durations = Vec::with_capacity(iterations);

    println!(
        "Running io-uring serverlessllm load for '{}' ({iterations}x)",
        label
    );
    for i in 0..iterations {
        let start = Instant::now();
        let model = serverlessllm::Model::load_io_uring(&dir)?;
        let elapsed = start.elapsed();
        durations.push(elapsed);
        let bytes: usize = (&model).into_iter().map(|(_, t)| t.data().len()).sum();
        println!(
            "  iteration {}: {} tensors, {} bytes, {:.2}ms",
            i + 1,
            model.len(),
            bytes,
            elapsed.as_secs_f64() * 1000.0
        );
        black_box((model.len(), bytes));
        if config.evict_page_cache {
            evict::evict_page_cache(&dir);
        }
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
        other => Err(ProfileError::new(format!("Unknown serverlessllm case '{}'", other)).into()),
    }
}
