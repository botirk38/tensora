use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor_store::formats::safetensors;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};
use crate::stats::summarize;

#[cfg(unix)]
fn drop_page_cache(path: &Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(file) = std::fs::File::open(path) {
        let fd = file.as_raw_fd();
        unsafe {
            let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
}

#[cfg(not(unix))]
fn drop_page_cache(_path: &Path) {}

fn drop_page_cache_for_dir(dir: &Path) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_file()
            {
                drop_page_cache(&entry.path());
            }
        }
    }
}

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

            let dir = entry.path();
            let has_safetensors = std::fs::read_dir(&dir)
                .ok()
                .into_iter()
                .flatten()
                .flatten()
                .any(|file| {
                    file.path()
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.ends_with(".safetensors"))
                });

            if has_safetensors {
                fixtures.push((entry.file_name().to_string_lossy().to_string(), dir));
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
    let fixtures = fixtures(config)?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let total_bytes = total_file_bytes(&dir)?;
        let mut durations = Vec::with_capacity(iterations);

        println!(
            "Running sync safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_dir(&dir);
            }
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
    }
    Ok(())
}

fn profile_load_mmap(config: &ProfileConfig) -> ProfileResult {
    let fixtures = fixtures(config)?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let total_bytes = total_file_bytes(&dir)?;
        let mut durations = Vec::with_capacity(iterations);

        println!(
            "Running mmap safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        for i in 0..iterations {
            if config.cold_cache && i == 0 {
                drop_page_cache_for_dir(&dir);
            }
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
    }
    Ok(())
}

fn profile_load_async(config: &ProfileConfig, default: bool) -> ProfileResult {
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;
    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let cache_label = if config.cold_cache { "cold" } else { "warm" };
        let total_bytes = total_file_bytes(&dir)?;
        let mut durations = Vec::with_capacity(iterations);
        let label = if default { "default" } else { "async" };

        println!(
            "Running {label} safetensors load for '{}' ({iterations}x {cache_label})",
            fixture
        );
        rt.block_on(async {
            for i in 0..iterations {
                if config.cold_cache && i == 0 {
                    drop_page_cache_for_dir(&dir);
                }
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
        other => Err(ProfileError::new(format!("Unknown safetensors case '{}'", other)).into()),
    }
}
