use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use ::safetensors::SafeTensors;
use tensor_store::formats::safetensors;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};

#[cfg(unix)]
fn drop_page_cache(path: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(file) = std::fs::File::open(path) {
        let fd = file.as_raw_fd();
        unsafe { libc::posix_madvise(fd as *mut libc::c_void, 0, libc::POSIX_MADV_DONTNEED) };
    }
}

#[cfg(not(unix))]
fn drop_page_cache(_path: &std::path::Path) {
    // No-op on non-unix
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "io-uring-load" => io_uring_load(config),
        "io-uring-parallel" => io_uring_parallel(config),
        "io-uring-prewarmed" => io_uring_prewarmed(config),
        "tokio-load" => tokio_load(config),
        "tokio-parallel" => tokio_parallel(config),
        "tokio-prewarmed" => tokio_prewarmed(config),
        "sync" => sync_load(config),
        "mmap" => mmap_load(config),
        "original" => original_load(config),
        other => Err(ProfileError::new(format!("Unknown safetensors case '{}'", other)).into()),
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

fn fixtures(config: &ProfileConfig) -> Result<Vec<(String, PathBuf)>, ProfileError> {
    let fixtures = discover_fixtures();
    if fixtures.is_empty() {
        return Err(ProfileError::new(
            "No safetensors fixtures found under 'fixtures/'.",
        ));
    }

    if let Some(name) = &config.fixture {
        let filtered: Vec<_> = fixtures
            .into_iter()
            .filter(|(fixture_name, _)| fixture_name == name)
            .collect();
        if filtered.is_empty() {
            return Err(ProfileError::new(format!(
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
fn io_uring_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running io_uring safetensors load for '{}' ({iterations}x)",
            fixture
        );
        tokio_uring::start(async {
            for _ in 0..iterations {
                let data = safetensors::Model::load(&path_str).await?;
                let tensor_count = data.names().len();
                black_box((data, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn io_uring_load(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new("io_uring safetensors cases are only available on Linux").into())
}

#[cfg(target_os = "linux")]
fn io_uring_parallel(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let chunks = num_cpus::get();

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running io_uring safetensors parallel load ({chunks} chunks) for '{}' ({iterations}x)",
            fixture
        );
        tokio_uring::start(async {
            for _ in 0..iterations {
                let data = safetensors::Model::load_parallel(&path_str, chunks).await?;
                let tensor_count = data.tensors().names().len();
                black_box((data, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn io_uring_parallel(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new("io_uring safetensors cases are only available on Linux").into())
}

#[cfg(target_os = "linux")]
fn io_uring_prewarmed(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running io_uring safetensors prewarmed load for '{}' ({iterations}x)",
            fixture
        );
        tokio_uring::start(async {
            for _ in 0..2 {
                let _ = safetensors::Model::load(&path_str).await?;
            }

            for _ in 0..iterations {
                let data = safetensors::Model::load(&path_str).await?;
                let tensor_count = data.names().len();
                black_box((data, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn io_uring_prewarmed(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new("io_uring safetensors cases are only available on Linux").into())
}

#[cfg(not(target_os = "linux"))]
fn tokio_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio safetensors load for '{}' ({iterations}x cold)",
            fixture
        );
        rt.block_on(async {
            for i in 0..iterations {
                if i == 0 {
                    drop_page_cache(std::path::Path::new(&path_str));
                }
                let start = Instant::now();
                let data = safetensors::Model::load(&path_str).await?;
                let elapsed = start.elapsed();
                let tensor_count = data.names().len();
                let bytes = data.into_bytes();
                println!(
                    "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                    i + 1,
                    tensor_count,
                    bytes.len(),
                    elapsed.as_secs_f64() * 1000.0
                );
                black_box((bytes, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn tokio_load(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new(
        "tokio safetensors cases are only compiled on non-Linux targets in this harness",
    )
    .into())
}

#[cfg(not(target_os = "linux"))]
fn tokio_parallel(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio safetensors parallel load (4 chunks) for '{}' ({iterations}x)",
            fixture
        );
        rt.block_on(async {
            for _ in 0..iterations {
                let data = safetensors::Model::load_parallel(&path_str, 4).await?;
                let tensor_count = data.tensors().names().len();
                black_box((data, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn tokio_parallel(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new(
        "tokio safetensors cases are only compiled on non-Linux targets in this harness",
    )
    .into())
}

#[cfg(not(target_os = "linux"))]
fn tokio_prewarmed(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio safetensors prewarmed load for '{}' ({iterations}x)",
            fixture
        );
        rt.block_on(async {
            for _ in 0..2 {
                let _ = safetensors::Model::load(&path_str).await?;
            }

            for _ in 0..iterations {
                let data = safetensors::Model::load(&path_str).await?;
                let tensor_count = data.names().len();
                black_box((data, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn tokio_prewarmed(_config: &ProfileConfig) -> ProfileResult {
    Err(ProfileError::new(
        "tokio safetensors cases are only compiled on non-Linux targets in this harness",
    )
    .into())
}

fn sync_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running sync safetensors load for '{}' ({iterations}x cold)",
            fixture
        );
        for i in 0..iterations {
            if i == 0 {
                drop_page_cache(&path);
            }
            let start = Instant::now();
            let data = safetensors::Model::load_sync(&path)?;
            let elapsed = start.elapsed();
            let tensor_count = data.names().len();
            let bytes = data.into_bytes();
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                bytes.len(),
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((bytes, tensor_count));
        }
    }

    Ok(())
}

fn mmap_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running mmap safetensors load for '{}' ({iterations}x cold)",
            fixture
        );
        for i in 0..iterations {
            if i == 0 {
                drop_page_cache(&path);
            }
            let start = Instant::now();
            let data = safetensors::MmapModel::load(&path)?;
            // Access all tensor data (simulates what Python conversion does)
            let mut bytes = 0;
            for name in data.tensors().names() {
                if let Ok(tensor) = data.tensors().tensor(name) {
                    bytes += tensor.data().len();
                }
            }
            let elapsed = start.elapsed();
            let tensor_count = data.tensors().names().len();
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((data, tensor_count));
        }
    }

    Ok(())
}

fn original_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running original safetensors load for '{}' ({iterations}x cold)",
            fixture
        );
        for i in 0..iterations {
            if i == 0 {
                drop_page_cache(&path);
            }
            let start = Instant::now();
            let bytes = fs::read(&path)?;
            let data = SafeTensors::deserialize(&bytes)?;
            let tensor_count = data.names().len();
            let elapsed = start.elapsed();
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                bytes.len(),
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((tensor_count, bytes.len()));
        }
    }

    Ok(())
}
