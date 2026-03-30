use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor_store::formats::serverlessllm;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};
use crate::stats::summarize;

#[cfg(unix)]
fn drop_page_cache_for_file(path: &Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(file) = std::fs::File::open(path) {
        let fd = file.as_raw_fd();
        unsafe {
            let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(unix)]
fn drop_page_cache_for_dir(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_file()
            {
                drop_page_cache_for_file(&entry.path());
            }
        }
    }
}

#[cfg(not(unix))]
fn drop_page_cache_for_file(_path: &Path) {}

#[cfg(not(unix))]
fn drop_page_cache_for_dir(_dir: &Path) {}

fn discover_fixtures() -> Vec<(String, PathBuf)> {
    let fixtures_dir = Path::new("fixtures");
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
                fixtures.push((entry.file_name().to_string_lossy().to_string(), model_dir));
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

fn profile_load_sync(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);

        println!(
            "Running sync serverlessllm load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_dir(&dir);
            }
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
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
    }
    Ok(())
}

fn profile_load_mmap(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);

        println!(
            "Running mmap serverlessllm load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_dir(&dir);
            }
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
        }

        if let Some(summary) = summarize(&durations) {
            println!(
                "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                summary.mean_ms, summary.min_ms, summary.max_ms
            );
        }
    }
    Ok(())
}

fn profile_load_async(config: &ProfileConfig, default: bool) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let mut durations = Vec::with_capacity(iterations);
        let label = if default { "default" } else { "async" };

        println!(
            "Running {label} serverlessllm load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        rt.block_on(async {
            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_dir(&dir);
                }
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
            }

            if let Some(summary) = summarize(&durations) {
                println!(
                    "  summary: mean {:.2}ms | min {:.2}ms | max {:.2}ms",
                    summary.mean_ms, summary.min_ms, summary.max_ms
                );
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }
    Ok(())
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "default" => profile_load_async(config, true),
        "sync" => profile_load_sync(config),
        "async" => profile_load_async(config, false),
        "mmap" => profile_load_mmap(config),
        other => Err(ProfileError::new(format!("Unknown serverlessllm case '{}'", other)).into()),
    }
}
