use std::hint::black_box;
use std::path::PathBuf;

use tensor_store::readers::serverlessllm;
use tensor_store::readers::traits::TensorMetadata;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};

const CASES: &[&str] = &["async-load", "sync-load", "mmap-load"];

pub const fn available_cases() -> &'static [&'static str] {
    CASES
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "async-load" => async_load(config),
        "sync-load" => sync_load(config),
        "mmap-load" => mmap_load(config),
        other => Err(ProfileError::new(format!(
            "Unknown serverlessllm case '{}'. Available: {}",
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

fn fixtures(config: &ProfileConfig) -> Result<Vec<(String, PathBuf)>, ProfileError> {
    let fixtures = discover_fixtures();
    if fixtures.is_empty() {
        return Err(ProfileError::new(
            "No serverlessllm fixtures found under 'fixtures/'.",
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
fn async_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let dir_str = dir
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running io_uring serverlessllm load for '{}' ({iterations}x)",
            fixture
        );
        tokio_uring::start(async {
            for _ in 0..iterations {
                let model = serverlessllm::load(&dir_str)?;
                let tensor_count = model.len();
                let mut total_bytes = 0;
                for (_name, tensor) in &model {
                    total_bytes += tensor.data().len();
                }
                black_box((total_bytes, tensor_count));
            }

            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn async_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let dir_str = dir
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio serverlessllm load for '{}' ({iterations}x)",
            fixture
        );
        rt.block_on(async {
            for _ in 0..iterations {
                let model = serverlessllm::load(&dir_str)?;
                let tensor_count = model.len();
                let mut total_bytes = 0;
                for (_name, tensor) in &model {
                    total_bytes += tensor.data().len();
                }
                black_box((total_bytes, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

fn sync_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running sync serverlessllm load for '{}' ({iterations}x)",
            fixture
        );
        for _ in 0..iterations {
            let model = serverlessllm::load(&dir)?;
            let tensor_count = model.len();
            let mut total_bytes = 0;
            for (_name, tensor) in &model {
                total_bytes += tensor.data().len();
            }
            black_box((total_bytes, tensor_count));
        }
    }

    Ok(())
}

fn mmap_load(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running mmap serverlessllm load for '{}' ({iterations}x)",
            fixture
        );
        for _ in 0..iterations {
            let model = serverlessllm::load_mmap(&dir)?;
            let tensor_count = model.len();
            let mut total_bytes = 0;
            for name in model.tensor_names() {
                let tensor = model.tensor(name).ok_or_else(|| {
                    ProfileError::new(format!("Missing tensor '{}' in mmap model", name))
                })?;
                total_bytes += tensor.data().len();
            }
            black_box((total_bytes, tensor_count));
        }
    }

    Ok(())
}
