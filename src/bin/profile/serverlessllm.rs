use std::hint::black_box;
use std::path::PathBuf;

use tensor_store::formats::serverlessllm;

use crate::config::{ProfileConfig, ProfileError, ProfileResult};

#[cfg(unix)]
fn drop_page_cache_for_dir(dir: &std::path::Path) {
    use std::os::unix::io::AsRawFd;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type()
                && file_type.is_file()
                && let Ok(file) = std::fs::File::open(entry.path())
            {
                let fd = file.as_raw_fd();
                unsafe { libc::posix_madvise(fd as *mut libc::c_void, 0, libc::POSIX_MADV_DONTNEED) };
            }
        }
    }
}

#[cfg(not(unix))]
fn drop_page_cache_for_dir(_dir: &std::path::Path) {
    // No-op on non-unix
}

pub fn run(case: &str, config: &ProfileConfig) -> ProfileResult {
    match case {
        "async-load" => async_load(config),
        "sync-load" => sync_load(config),
        "mmap-load" => mmap_load(config),
        other => Err(ProfileError::new(format!("Unknown serverlessllm case '{}'", other)).into()),
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
            "Running io_uring serverlessllm load for '{}' ({iterations}x cold)",
            fixture
        );
        tokio_uring::start(async {
            for i in 0..iterations {
                if i == 0 {
                    drop_page_cache_for_dir(std::path::Path::new(&dir_str));
                }
                let model = serverlessllm::Model::load(&dir_str).await?;
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
    use std::time::Instant;
    let fixtures = fixtures(config)?;
    let rt = tokio::runtime::Runtime::new()?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        let dir_str = dir
            .to_str()
            .ok_or_else(|| ProfileError::new("Fixture path contains invalid UTF-8"))?
            .to_owned();

        println!(
            "Running tokio serverlessllm load for '{}' ({iterations}x cold)",
            fixture
        );
        rt.block_on(async {
            for i in 0..iterations {
                if i == 0 {
                    drop_page_cache_for_dir(std::path::Path::new(&dir_str));
                }
                let start = Instant::now();
                let model = serverlessllm::Model::load(&dir_str).await?;
                let elapsed = start.elapsed();
                let tensor_count = model.len();
                let mut total_bytes = 0;
                for (_name, tensor) in &model {
                    total_bytes += tensor.data().len();
                }
                println!(
                    "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                    i + 1,
                    tensor_count,
                    total_bytes,
                    elapsed.as_secs_f64() * 1000.0
                );
                black_box((total_bytes, tensor_count));
            }
            Ok::<_, tensor_store::ReaderError>(())
        })?;
    }

    Ok(())
}

fn sync_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running sync serverlessllm load for '{}' ({iterations}x cold)",
            fixture
        );
        for i in 0..iterations {
            if i == 0 {
                drop_page_cache_for_dir(&dir);
            }
            let start = Instant::now();
            let model = serverlessllm::Model::load_sync(&dir)?;
            let elapsed = start.elapsed();
            let tensor_count = model.len();
            let mut total_bytes = 0;
            for (_name, tensor) in &model {
                total_bytes += tensor.data().len();
            }
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((total_bytes, tensor_count));
        }
    }

    Ok(())
}

fn mmap_load(config: &ProfileConfig) -> ProfileResult {
    use std::time::Instant;
    let fixtures = fixtures(config)?;

    for (fixture, dir) in fixtures {
        let iterations = config.normalized_iterations();
        println!(
            "Running mmap serverlessllm load for '{}' ({iterations}x cold)",
            fixture
        );
        for i in 0..iterations {
            if i == 0 {
                drop_page_cache_for_dir(&dir);
            }
            let start = Instant::now();
            let model = serverlessllm::MmapModel::load(&dir)?;
            let elapsed = start.elapsed();
            let tensor_count = model.len();
            let mut total_bytes = 0;
            for name in model.tensor_names() {
                let tensor = model.tensor(name).ok_or_else(|| {
                    ProfileError::new(format!("Missing tensor '{}' in mmap model", name))
                })?;
                total_bytes += tensor.data().len();
            }
            println!(
                "  iteration {}: {} tensors, {} bytes, {:.2}ms",
                i + 1,
                tensor_count,
                total_bytes,
                elapsed.as_secs_f64() * 1000.0
            );
            black_box((total_bytes, tensor_count));
        }
    }

    Ok(())
}
