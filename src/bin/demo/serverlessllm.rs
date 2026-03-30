use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor_store::formats::serverlessllm;

use crate::config::{DemoConfig, DemoError, DemoResult, format_bytes};

pub fn run(scenario: &str, config: &DemoConfig) -> DemoResult {
    match scenario {
        "async" => demo_async(config),
        "sync" => demo_sync(config),
        "mmap" => demo_mmap(config),
        "metadata" => demo_metadata(config),
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

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = std::path::Path::new("fixtures");
    let mut fixtures = Vec::new();

    if let Ok(entries) = std::fs::read_dir(fixtures_dir) {
        for entry in entries.flatten() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if !file_type.is_dir() {
                continue;
            }

            let model_dir = entry.path().join("model_serverlessllm");
            if model_dir.exists() && model_dir.is_dir() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                fixtures.push((model_name, model_dir));
            }
        }
    }

    fixtures.sort_by(|a, b| a.0.cmp(&b.0));
    fixtures
}

fn fixtures(config: &DemoConfig) -> Result<Vec<(String, PathBuf)>, DemoError> {
    let fixtures = discover_fixtures();
    if fixtures.is_empty() {
        return Err(DemoError::new(
            "No serverlessllm fixtures found under 'fixtures/'.",
        ));
    }

    if let Some(name) = &config.fixture {
        let filtered: Vec<_> = fixtures
            .into_iter()
            .filter(|(fixture_name, _)| fixture_name == name)
            .collect();
        if filtered.is_empty() {
            return Err(DemoError::new(format!(
                "Fixture '{}' not found. Available: {}",
                name,
                discover_fixtures()
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(filtered)
    } else {
        Ok(fixtures)
    }
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

#[cfg(target_os = "linux")]
fn demo_async(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Async Sequential Loading Demo (tokio) ===\n");

    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(async {
        for (name, dir) in fixtures {
            println!("Fixture: {}", name);

            let total_size_bytes = total_size(&dir)?;
            let partition_count = count_partitions(&dir);

            println!("  Directory: model_serverlessllm/");
            println!("  Size: {}", format_bytes(total_size_bytes));
            println!("  Partitions: {}", partition_count);

            let io_before = crate::io_metrics::capture_disk_snapshot().ok();

            let start = Instant::now();
            let model = serverlessllm::Model::load(&dir).await?;
            let duration = start.elapsed();

            println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

            let tensor_count = model.len();
            println!("  Tensors: {}", tensor_count);

            let throughput = total_size_bytes as f64 / duration.as_secs_f64() / 1e9;
            println!("  Throughput: {:.2} GB/s", throughput);

            crate::io_metrics::display_io_metrics_delta(io_before, duration);
            println!();
        }

        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

fn demo_sync(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Sync Loading Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, dir) in fixtures {
        println!("Fixture: {}", name);

        let total_size_bytes = total_size(&dir)?;
        let partition_count = count_partitions(&dir);

        println!("  Directory: model_serverlessllm/");
        println!("  Size: {}", format_bytes(total_size_bytes));
        println!("  Partitions: {}", partition_count);

        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let model = serverlessllm::Model::load_sync(&dir)?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = model.len();
        println!("  Tensors: {}", tensor_count);

        let throughput = total_size_bytes as f64 / duration.as_secs_f64() / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();
    }

    Ok(())
}

fn demo_mmap(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Memory-Mapped Loading Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, dir) in fixtures {
        println!("Fixture: {}", name);

        let total_size_bytes = total_size(&dir)?;
        let partition_count = count_partitions(&dir);

        println!("  Directory: model_serverlessllm/");
        println!("  Size: {}", format_bytes(total_size_bytes));
        println!("  Partitions: {}", partition_count);

        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let model = serverlessllm::MmapModel::open(&dir)?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = model.len();
        println!("  Tensors: {}", tensor_count);

        let throughput = total_size_bytes as f64 / duration.as_secs_f64() / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();
    }

    Ok(())
}

fn demo_metadata(config: &DemoConfig) -> DemoResult {
    println!("=== ServerlessLLM Metadata Exploration Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, dir) in fixtures {
        println!("Fixture: {}", name);

        let total_size_bytes = total_size(&dir)?;
        let partition_count = count_partitions(&dir);

        println!("  Directory: model_serverlessllm/");
        println!("  Size: {}", format_bytes(total_size_bytes));
        println!("  Partitions: {}", partition_count);

        let model = serverlessllm::Model::load_sync(&dir)?;
        let tensor_names = model.tensor_names();

        println!("  Tensors: {}\n", tensor_names.len());

        println!("  Sample tensors (first 10):");
        for name in tensor_names.iter().take(10) {
            if let Some(tensor) = model.tensor(name) {
                println!(
                    "    - {}: {:?} ({}) - {}",
                    name,
                    tensor.shape(),
                    tensor.dtype(),
                    format_bytes(tensor.size() as u64)
                );
            }
        }
        println!();
    }

    Ok(())
}
