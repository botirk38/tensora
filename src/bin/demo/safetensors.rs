//! SafeTensors demo scenarios.

use std::path::{Path, PathBuf};
use std::time::Instant;

use tensora::TensorView;
use tensora::formats::safetensors;

use crate::config::{DemoConfig, DemoError, DemoResult, format_bytes};

fn resolve_dir(config: &DemoConfig) -> Result<PathBuf, DemoError> {
    tensora::hf_model::ensure_safetensors_hub_dir(&config.model_id)
        .map_err(|e| DemoError::new(e.to_string()))
}

fn dir_safetensors_total_bytes(dir: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "safetensors") && path.is_file() {
            total += path.metadata()?.len();
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
        other => Err(DemoError::new(format!("Unknown safetensors scenario '{}'", other)).into()),
    }
}

fn demo_async(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Async Sequential Loading Demo (tokio) ===\n");

    let path = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);
    let file_size = dir_safetensors_total_bytes(&path)?;
    println!("  Hub snapshot dir: {}", path.display());
    println!("  Total safetensors size: {}", format_bytes(file_size));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(async {
        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let data = safetensors::Model::load(&path).await?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = data.tensor_names().len();
        println!("  Tensors: {}", tensor_count);

        let throughput = file_size as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();

        Ok::<_, tensora::ReaderError>(())
    })?;

    Ok(())
}

fn demo_sync(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Sync Loading Demo ===\n");

    let path = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);
    let file_size = dir_safetensors_total_bytes(&path)?;
    println!("  Hub snapshot dir: {}", path.display());
    println!("  Total safetensors size: {}", format_bytes(file_size));

    let io_before = crate::io_metrics::capture_disk_snapshot().ok();

    let start = Instant::now();
    let data = safetensors::Model::load_sync(&path)?;
    let duration = start.elapsed();

    println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

    let tensor_count = data.tensor_names().len();
    println!("  Tensors: {}", tensor_count);

    let throughput = file_size as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
    println!("  Throughput: {:.2} GB/s", throughput);

    crate::io_metrics::display_io_metrics_delta(io_before, duration);
    println!();

    Ok(())
}

fn demo_mmap(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Memory-Mapped Loading Demo ===\n");

    let path = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);
    let file_size = dir_safetensors_total_bytes(&path)?;
    println!("  Hub snapshot dir: {}", path.display());
    println!("  Total safetensors size: {}", format_bytes(file_size));

    let io_before = crate::io_metrics::capture_disk_snapshot().ok();

    let start = Instant::now();
    let data = safetensors::MmapModel::open(&path)?;
    let duration = start.elapsed();

    println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

    let tensor_count = data.tensor_names().len();
    println!("  Tensors: {}", tensor_count);

    let throughput = file_size as f64 / duration.as_secs_f64().max(1e-9) / 1e9;
    println!("  Throughput: {:.2} GB/s", throughput);

    crate::io_metrics::display_io_metrics_delta(io_before, duration);
    println!();

    Ok(())
}

fn demo_metadata(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Metadata Exploration Demo ===\n");

    let path = resolve_dir(config)?;
    let name = &config.model_id;
    println!("Model: {}", name);
    let file_size = dir_safetensors_total_bytes(&path)?;
    println!("  Hub snapshot dir: {}", path.display());
    println!("  Total safetensors size: {}", format_bytes(file_size));

    let data = safetensors::Model::load_sync(&path)?;
    let names = data.tensor_names();

    println!("  Tensors: {}\n", names.len());

    println!("  Sample tensors (first 10):");
    for tensor_name in names.iter().take(10) {
        if let Ok(tensor) = data.tensor(tensor_name) {
            println!(
                "    - {}: {:?} ({:?}) - {}",
                tensor_name,
                tensor.shape(),
                tensor.dtype(),
                format_bytes(u64::try_from(tensor.data().len()).unwrap_or(u64::MAX))
            );
        }
    }
    println!();

    Ok(())
}
