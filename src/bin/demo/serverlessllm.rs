//! ServerlessLLM demo scenarios.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tensora::formats::serverlessllm;

use crate::config::{DemoConfig, DemoError, DemoResult, format_bytes};

fn resolve_dir(config: &DemoConfig) -> Result<PathBuf, DemoError> {
    let safetensors_dir = tensora::hf_model::ensure_safetensors_hub_dir(&config.model_id)
        .map_err(|e| DemoError::new(e.to_string()))?;
    tensora::hf_model::ensure_serverlessllm_cache_dir(&config.model_id, &safetensors_dir)
        .map_err(|e| DemoError::new(e.to_string()))
}

fn count_partitions(dir: &Path) -> usize {
    let mut count = 0;
    let mut partition_id = 0;
    loop {
        let partition_path = dir.join(format!("tensor.data_{}", partition_id));
        if partition_path.exists() {
            count += 1;
            partition_id += 1;
        } else {
            break;
        }
    }
    count
}

fn total_size(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0;
    let mut partition_id = 0;
    loop {
        let partition_path = dir.join(format!("tensor.data_{}", partition_id));
        if partition_path.exists() {
            total += std::fs::metadata(&partition_path)?.len();
            partition_id += 1;
        } else {
            break;
        }
    }
    Ok(total)
}

pub fn run(scenario: &str, config: &DemoConfig) -> DemoResult {
    match scenario {
        "async" => demo_async(config),
        "sync" => demo_sync(config),
        "mmap" => demo_mmap(config),
        "metadata" => demo_metadata(config),
        "parallel-sync" => {
            Err(DemoError::new("parallel-sync demo is not wired for Hub-backed models yet").into())
        }
        "all" => {
            demo_async(config)?;
            println!();
            demo_sync(config)?;
            println!();
            demo_mmap(config)?;
            println!();
            demo_metadata(config)?;
            Ok(())
        }
        other => Err(DemoError::new(format!("Unknown serverlessllm scenario '{}'", other)).into()),
    }
}

fn demo_async(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Async Sequential Loading Demo (tokio) ===\n");

    let dir = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);

    let total_size_bytes = total_size(&dir)?;
    let partition_count = count_partitions(&dir);

    println!("  Directory: {}", dir.display());
    println!("  Size: {}", format_bytes(total_size_bytes));
    println!("  Partitions: {}", partition_count);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(async {
        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let model = serverlessllm::Model::load(&dir).await?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = model.len();
        println!("  Tensors: {}", tensor_count);

        let throughput = total_size_bytes as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();

        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

fn demo_sync(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Sync Loading Demo ===\n");

    let dir = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);

    let total_size_bytes = total_size(&dir)?;
    let partition_count = count_partitions(&dir);

    println!("  Directory: {}", dir.display());
    println!("  Size: {}", format_bytes(total_size_bytes));
    println!("  Partitions: {}", partition_count);

    let io_before = crate::io_metrics::capture_disk_snapshot().ok();

    let start = Instant::now();
    let model = serverlessllm::Model::load_sync(&dir)?;
    let duration = start.elapsed();

    println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

    let tensor_count = model.len();
    println!("  Tensors: {}", tensor_count);

    let throughput = total_size_bytes as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
    println!("  Throughput: {:.2} GB/s", throughput);

    crate::io_metrics::display_io_metrics_delta(io_before, duration);
    println!();

    Ok(())
}

fn demo_mmap(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Memory-Mapped Loading Demo ===\n");

    let dir = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);

    let total_size_bytes = total_size(&dir)?;
    let partition_count = count_partitions(&dir);

    println!("  Directory: {}", dir.display());
    println!("  Size: {}", format_bytes(total_size_bytes));
    println!("  Partitions: {}", partition_count);

    let io_before = crate::io_metrics::capture_disk_snapshot().ok();

    let start = Instant::now();
    let model = serverlessllm::MmapModel::open(&dir)?;
    let duration = start.elapsed();

    println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

    let tensor_count = model.len();
    println!("  Tensors: {}", tensor_count);

    let throughput = total_size_bytes as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
    println!("  Throughput: {:.2} GB/s", throughput);

    crate::io_metrics::display_io_metrics_delta(io_before, duration);
    println!();

    Ok(())
}

fn demo_metadata(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Metadata Exploration Demo ===\n");

    let dir = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);

    let total_size_bytes = total_size(&dir)?;
    let partition_count = count_partitions(&dir);

    println!("  Directory: {}", dir.display());
    println!("  Size: {}", format_bytes(total_size_bytes));
    println!("  Partitions: {}", partition_count);

    let model = serverlessllm::Model::load_sync(&dir)?;
    let tensor_names = model.tensor_names();

    println!("  Tensors: {}\n", tensor_names.len());

    println!("  Sample tensors (first 10):");
    for n in tensor_names.iter().take(10) {
        if let Some(tensor) = model.tensor(n) {
            println!(
                "    - {}: {:?} ({}) - {}",
                n,
                tensor.shape(),
                tensor.dtype(),
                format_bytes(tensor.size() as u64)
            );
        }
    }
    println!();

    Ok(())
}
