//! ServerlessLLM Reader Profiling Harness
//!
//! This profiling harness provides detailed performance analysis for serverlessllm reader operations.
//! Use with profiling tools like flamegraph, perf, or valgrind for deep performance insights.
//!
//! ## Usage Examples
//!
//! ### Flamegraph Profiling
//! ```bash
//! cargo flamegraph --bin serverlessllm_reader -- --profile async
//! cargo flamegraph --bin serverlessllm_reader -- --profile sync
//! ```
//!
//! ### Perf Profiling
//! ```bash
//! perf record -g cargo run --release --bin serverlessllm_reader -- --profile async
//! perf report
//! ```
//!
//! ### Valgrind Profiling
//! ```bash
//! valgrind --tool=callgrind cargo run --release --bin serverlessllm_reader -- --profile sync
//! kcachegrind callgrind.out.*
//! ```
//!
//! ## Test Data
//!
//! Requires a `test_model_serverlessllm/` directory containing:
//! - `tensor_index.json` - ServerlessLLM index file
//! - `tensor.data_0`, `tensor.data_1`, ... - Partitioned tensor data files
//!   For realistic profiling, use a large model with multiple data files.

use std::hint::black_box;
use std::time::Instant;
use tensor_store::readers::serverlessllm;
use tensor_store::readers::traits::TensorMetadata;

#[derive(Debug)]
enum ProfileMode {
    Async,
    Sync,
}

impl ProfileMode {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "async" => Ok(Self::Async),
            "sync" => Ok(Self::Sync),
            _ => Err(format!("Unknown profile mode: {}. Use async or sync", s)),
        }
    }
}

fn get_fixture() -> String {
    let args: Vec<String> = std::env::args().collect();
    let fixture_index = args.iter().position(|arg| arg == "--fixture");
    fixture_index.map_or_else(
        || "large".to_string(),
        |idx| {
            if idx + 1 < args.len() {
                args[idx + 1].clone()
            } else {
                "large".to_string()
            }
        },
    )
}

fn get_profile_mode() -> Result<ProfileMode, String> {
    let args: Vec<String> = std::env::args().collect();
    let profile_index = args.iter().position(|arg| arg == "--profile");
    profile_index.map_or_else(
        || Err("Missing --profile argument".to_string()),
        |idx| {
            if idx + 1 < args.len() {
                ProfileMode::from_str(&args[idx + 1])
            } else {
                Err("Missing profile mode after --profile".to_string())
            }
        },
    )
}

async fn profile_async(serverlessllm_dir: &str) {
    println!("🔥 Profiling async serverlessllm reader...");

    // Warmup phase - populate buffer pools and caches
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = serverlessllm::load(serverlessllm_dir).await.unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase - single iteration to avoid memory accumulation
    println!("  Starting profiling...");
    let start = Instant::now();
    let model = serverlessllm::load(black_box(serverlessllm_dir))
        .await
        .unwrap();
    let load_time = start.elapsed();

    let tensor_count = model.len();
    let mut bytes = 0;
    for (_name, tensor) in &model {
        bytes += tensor.data().len();
    }

    let avg_time = load_time;
    let avg_tensors = tensor_count;
    let avg_bytes = bytes;

    println!("\n📊 Async Profiling Results:");
    println!("  Average load time: {:.2}ms", avg_time.as_millis());
    println!("  Average tensors: {}", avg_tensors);
    println!("  Average bytes: {}", avg_bytes);
    println!("  Total profiling time: {:.2}s", load_time.as_secs_f64());
}

fn profile_sync(serverlessllm_dir: &str) {
    println!("🔥 Profiling sync serverlessllm reader...");

    // Warmup phase
    println!("  Warming up...");
    for i in 0..3 {
        let _warmup = serverlessllm::load_sync(serverlessllm_dir).unwrap();
        println!("    Warmup iteration {} complete", i + 1);
    }

    // Profiling phase
    println!("  Starting profiling...");
    let start = Instant::now();
    let model = serverlessllm::load_sync(black_box(serverlessllm_dir)).unwrap();
    let load_time = start.elapsed();

    let tensor_count = model.len();
    let mut bytes = 0;
    for (_name, tensor) in &model {
        bytes += tensor.data().len();
    }

    let avg_time = load_time;
    let avg_tensors = tensor_count;
    let avg_bytes = bytes;

    println!(
        "    Single iteration: {} tensors, {} bytes in {:.2}ms",
        tensor_count,
        bytes,
        load_time.as_millis()
    );

    println!("\n📊 Sync Profiling Results:");
    println!("  Load time: {:.2}ms", avg_time.as_millis());
    println!("  Tensors: {}", avg_tensors);
    println!("  Bytes: {}", avg_bytes);
    println!("  Total profiling time: {:.2}s", load_time.as_secs_f64());
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = get_fixture();
    let mode = get_profile_mode().map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let serverlessllm_dir = format!("tests/fixtures/{}/tiny_serverlessllm", fixture);

    // Check if test directory exists
    if !std::path::Path::new(&serverlessllm_dir).exists() {
        eprintln!("❌ Test directory '{}' not found", serverlessllm_dir);
        eprintln!("Please ensure the fixture exists");
        std::process::exit(1);
    }

    println!("🚀 Starting serverlessllm reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Fixture: {}", fixture);
    println!("  Test directory: {}", serverlessllm_dir);
    println!();

    // Use minimal queue size for profiling to avoid memory conflicts with perf/flamegraph
    let mut binding = tokio_uring::builder();
    let builder = binding.entries(8);
    match builder.start(async {
        match mode {
            ProfileMode::Async => {
                profile_async(&serverlessllm_dir).await;
            }
            ProfileMode::Sync => {
                profile_sync(&serverlessllm_dir);
            }
        }

        println!("\n✅ Profiling complete!");
        Ok::<(), std::io::Error>(())
    }) {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("❌ Failed to initialize io_uring runtime: {}", e);
            eprintln!();
            eprintln!("💡 Troubleshooting:");
            eprintln!("  • If using flamegraph/perf: Try running without profiling tools first");
            eprintln!("  • If running as non-root: perf may be limited by memory locks");
            eprintln!(
                "  • Try increasing perf memory limits: sudo sysctl kernel.perf_event_mlock_kb=2048"
            );
            eprintln!(
                "  • Or run with: CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --bin serverlessllm_reader -- --profile sync"
            );
            std::process::exit(1);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = get_fixture();
    let mode = get_profile_mode().map_err(|e| {
        eprintln!("{}", e);
        std::process::exit(1);
    })?;

    let serverlessllm_dir = format!("tests/fixtures/{}/tiny_serverlessllm", fixture);

    // Check if test directory exists
    if !std::path::Path::new(&serverlessllm_dir).exists() {
        eprintln!("❌ Test directory '{}' not found", serverlessllm_dir);
        eprintln!("Please ensure the fixture exists");
        std::process::exit(1);
    }

    println!("🚀 Starting serverlessllm reader profiling");
    println!("  Mode: {:?}", mode);
    println!("  Fixture: {}", fixture);
    println!("  Test directory: {}", serverlessllm_dir);
    println!();
    rt.block_on(async {
        match mode {
            ProfileMode::Async => {
                profile_async(&serverlessllm_dir).await;
            }
            ProfileMode::Sync => {
                profile_sync(&serverlessllm_dir);
            }
        }

        println!("\n✅ Profiling complete!");
    });

    Ok(())
}
