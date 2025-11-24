use std::fs;
use std::hint::black_box;
use std::path::PathBuf;

use safetensors::SafeTensors;
use tensor_store::readers::safetensors;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};

const CASES: &[&str] = &[
    "io-uring-load",
    "io-uring-parallel",
    "io-uring-prewarmed",
    "tokio-load",
    "tokio-parallel",
    "tokio-prewarmed",
    "sync",
    "mmap",
    "original",
];

pub const fn available_cases() -> &'static [&'static str] {
    CASES
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
        other => Err(ProfileError::new(format!(
            "Unknown safetensors case '{}'. Available: {}",
            other,
            CASES.join(", ")
        ))
        .into()),
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
                let data = safetensors::load(&path_str).await?;
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
                let data = safetensors::load_parallel(&path_str, chunks).await?;
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
                let _ = safetensors::load(&path_str).await?;
            }

            for _ in 0..iterations {
                let data = safetensors::load(&path_str).await?;
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
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        let path_str = path
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio safetensors load for '{}' ({iterations}x)",
            fixture
        );
        rt.block_on(async {
            for _ in 0..iterations {
                let data = safetensors::load(&path_str).await?;
                let tensor_count = data.names().len();
                black_box((data, tensor_count));
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
                let data = safetensors::load_parallel(&path_str, 4).await?;
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
                let _ = safetensors::load(&path_str).await?;
            }

            for _ in 0..iterations {
                let data = safetensors::load(&path_str).await?;
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
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running sync safetensors load for '{}' ({iterations}x)",
            fixture
        );
        for _ in 0..iterations {
            let data = safetensors::load_sync(&path)?;
            let tensor_count = data.names().len();
            black_box((data.into_bytes(), tensor_count));
        }
    }

    Ok(())
}

fn mmap_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running mmap safetensors load for '{}' ({iterations}x)",
            fixture
        );
        for _ in 0..iterations {
            let data = safetensors::load_mmap(&path)?;
            let tensor_count = data.tensors().names().len();
            black_box((data, tensor_count));
        }
    }

    Ok(())
}

fn original_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, path) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running original safetensors load for '{}' ({iterations}x)",
            fixture
        );
        for _ in 0..iterations {
            let bytes = fs::read(&path)?;
            let data = SafeTensors::deserialize(&bytes)?;
            let tensor_count = data.names().len();
            black_box((tensor_count, bytes.len()));
        }
    }

    Ok(())
}
