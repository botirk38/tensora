use std::path::PathBuf;
use std::time::Instant;

use tensor_store::formats::safetensors;

use crate::config::{DemoConfig, DemoError, DemoResult, format_bytes};

pub fn run(scenario: &str, config: &DemoConfig) -> DemoResult {
    match scenario {
        "async" => demo_async(config),
        "sync" => demo_sync(config),
        "mmap" => demo_mmap(config),
        "parallel-async" => demo_parallel_async(config),
        "parallel-sync" => demo_parallel_sync(config),
        "metadata" => demo_metadata(config),
        "all" => {
            demo_async(config)?;
            println!();
            demo_sync(config)?;
            println!();
            demo_mmap(config)?;
            println!();
            demo_parallel_async(config)?;
            println!();
            demo_parallel_sync(config)?;
            println!();
            demo_metadata(config)?;
            Ok(())
        }
        other => Err(DemoError::new(format!("Unknown safetensors scenario '{}'", other)).into()),
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

            let model_path = entry.path().join("model.safetensors");
            if model_path.exists() {
                let model_name = entry.file_name().to_string_lossy().to_string();
                fixtures.push((model_name, model_path));
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
            "No safetensors fixtures found under 'fixtures/'.",
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

#[cfg(target_os = "linux")]
fn demo_async(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Async Sequential Loading Demo (io_uring) ===\n");

    let fixtures = fixtures(config)?;

    tokio_uring::start(async {
        for (name, path) in fixtures {
            println!("Fixture: {}", name);

            let file_size = std::fs::metadata(&path)?.len();
            println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
            println!("  Size: {}", format_bytes(file_size));

            let io_before = crate::io_metrics::capture_disk_snapshot().ok();

            let start = Instant::now();
            let data = safetensors::load(&path).await?;
            let duration = start.elapsed();

            println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

            let tensor_count = data.names().len();
            println!("  Tensors: {}", tensor_count);

            let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
            println!("  Throughput: {:.2} GB/s", throughput);

            crate::io_metrics::display_io_metrics_delta(io_before, duration);
            println!();
        }

        Ok::<_, tensor_store::ReaderError>(())
    })?;

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn demo_async(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Async Sequential Loading Demo (tokio) ===\n");

    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(async {
        for (name, path) in fixtures {
            println!("Fixture: {}", name);

            let file_size = std::fs::metadata(&path)?.len();
            println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
            println!("  Size: {}", format_bytes(file_size));

            let start = Instant::now();
            let data = safetensors::load(&path).await?;
            let duration = start.elapsed();

            println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

            let tensor_count = data.names().len();
            println!("  Tensors: {}", tensor_count);

            let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
            println!("  Throughput: {:.2} GB/s\n", throughput);
        }

        Ok::<_, tensor_store::ReaderError>(())
    })?;

    Ok(())
}

fn demo_sync(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Sync Loading Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, path) in fixtures {
        println!("Fixture: {}", name);

        let file_size = std::fs::metadata(&path)?.len();
        println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
        println!("  Size: {}", format_bytes(file_size));

        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let data = safetensors::load_sync(&path)?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = data.names().len();
        println!("  Tensors: {}", tensor_count);

        let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        // Always attempt to display IO metrics, even if minimal activity
        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();
    }

    Ok(())
}

fn demo_mmap(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Memory-Mapped Loading Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, path) in fixtures {
        println!("Fixture: {}", name);

        let file_size = std::fs::metadata(&path)?.len();
        println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
        println!("  Size: {}", format_bytes(file_size));

        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let data = safetensors::load_mmap(&path)?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = data.tensors().names().len();
        println!("  Tensors: {}", tensor_count);

        let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn demo_parallel_async(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Async Parallel Loading Demo (io_uring) ===\n");

    let fixtures = fixtures(config)?;
    let chunks = num_cpus::get();

    tokio_uring::start(async {
        for (name, path) in fixtures {
            println!("Fixture: {}", name);

            let file_size = std::fs::metadata(&path)?.len();
            println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
            println!("  Size: {}", format_bytes(file_size));
            println!("  Chunks: {}", chunks);

            let io_before = crate::io_metrics::capture_disk_snapshot().ok();

            let start = Instant::now();
            let data = safetensors::load_parallel(&path, chunks).await?;
            let duration = start.elapsed();

            println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

            let tensor_count = data.tensors().names().len();
            println!("  Tensors: {}", tensor_count);

            let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
            println!("  Throughput: {:.2} GB/s", throughput);

            crate::io_metrics::display_io_metrics_delta(io_before, duration);
            println!();
        }

        Ok::<_, tensor_store::ReaderError>(())
    })?;

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn demo_parallel_async(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Async Parallel Loading Demo (tokio) ===\n");

    let fixtures = fixtures(config)?;
    let chunks = num_cpus::get();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .build()?;

    rt.block_on(async {
        for (name, path) in fixtures {
            println!("Fixture: {}", name);

            let file_size = std::fs::metadata(&path)?.len();
            println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
            println!("  Size: {}", format_bytes(file_size));
            println!("  Chunks: {}", chunks);

            let start = Instant::now();
            let data = safetensors::load_parallel(&path, chunks).await?;
            let duration = start.elapsed();

            println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

            let tensor_count = data.tensors().names().len();
            println!("  Tensors: {}", tensor_count);

            let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
            println!("  Throughput: {:.2} GB/s\n", throughput);
        }

        Ok::<_, tensor_store::ReaderError>(())
    })?;

    Ok(())
}

fn demo_parallel_sync(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Sync Parallel Loading Demo ===\n");

    let fixtures = fixtures(config)?;
    let chunks = num_cpus::get();

    for (name, path) in fixtures {
        println!("Fixture: {}", name);

        let file_size = std::fs::metadata(&path)?.len();
        println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
        println!("  Size: {}", format_bytes(file_size));
        println!("  Chunks: {}", chunks);

        let io_before = crate::io_metrics::capture_disk_snapshot().ok();

        let start = Instant::now();
        let data = safetensors::load_parallel_sync(&path, chunks)?;
        let duration = start.elapsed();

        println!("  Loaded in: {:.2}ms", duration.as_secs_f64() * 1000.0);

        let tensor_count = data.names().len();
        println!("  Tensors: {}", tensor_count);

        let throughput = file_size as f64 / duration.as_secs_f64() / 1e9;
        println!("  Throughput: {:.2} GB/s", throughput);

        crate::io_metrics::display_io_metrics_delta(io_before, duration);
        println!();
    }

    Ok(())
}

fn demo_metadata(config: &DemoConfig) -> DemoResult {
    println!("=== SafeTensors Metadata Exploration Demo ===\n");

    let fixtures = fixtures(config)?;

    for (name, path) in fixtures {
        println!("Fixture: {}", name);

        let file_size = std::fs::metadata(&path)?.len();
        println!("  File: {}", path.file_name().unwrap().to_str().unwrap());
        println!("  Size: {}", format_bytes(file_size));

        let data = safetensors::load_sync(&path)?;
        let tensors = data.tensors();
        let names = tensors.names();

        println!("  Tensors: {}\n", names.len());

        println!("  Sample tensors (first 10):");
        for name in names.iter().take(10) {
            if let Ok(tensor) = tensors.tensor(name) {
                println!(
                    "    - {}: {:?} ({:?}) - {}",
                    name,
                    tensor.shape(),
                    tensor.dtype(),
                    format_bytes(u64::try_from(tensor.data().len()).unwrap_or(u64::MAX))
                );
            }
        }
        println!();
    }

    Ok(())
}
