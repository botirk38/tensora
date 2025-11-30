use clap::{Parser, Subcommand, ValueEnum};

mod config;
mod safetensors;
mod serverlessllm;

use config::ProfileConfig;

#[derive(Parser)]
#[command(name = "profile")]
#[command(about = "Profiling harness for tensor_store (no Criterion)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile SafeTensors loader
    Safetensors {
        /// Profiling case to run
        #[arg(value_enum)]
        case: SafeTensorsCase,

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,
    },
    /// Profile ServerlessLLM loader
    Serverlessllm {
        /// Profiling case to run
        #[arg(value_enum)]
        case: ServerlessLLMCase,

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsCase {
    /// io_uring async load (Linux only)
    IoUringLoad,
    /// io_uring parallel load (Linux only)
    IoUringParallel,
    /// io_uring prewarmed load (Linux only)
    IoUringPrewarmed,
    /// Tokio async load
    TokioLoad,
    /// Tokio parallel load
    TokioParallel,
    /// Tokio prewarmed load
    TokioPrewarmed,
    /// Synchronous load
    Sync,
    /// Memory-mapped load
    Mmap,
    /// Original safetensors crate load
    Original,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMCase {
    /// Async load
    AsyncLoad,
    /// Synchronous load
    SyncLoad,
    /// Memory-mapped load
    MmapLoad,
}

impl SafeTensorsCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::IoUringLoad => "io-uring-load",
            Self::IoUringParallel => "io-uring-parallel",
            Self::IoUringPrewarmed => "io-uring-prewarmed",
            Self::TokioLoad => "tokio-load",
            Self::TokioParallel => "tokio-parallel",
            Self::TokioPrewarmed => "tokio-prewarmed",
            Self::Sync => "sync",
            Self::Mmap => "mmap",
            Self::Original => "original",
        }
    }
}

impl ServerlessLLMCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::AsyncLoad => "async-load",
            Self::SyncLoad => "sync-load",
            Self::MmapLoad => "mmap-load",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Safetensors {
            case,
            fixture,
            iterations,
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
            };
            safetensors::run(case.as_str(), &config)?;
        }
        Commands::Serverlessllm {
            case,
            fixture,
            iterations,
        } => {
            let config = ProfileConfig {
                iterations,
                fixture,
            };
            serverlessllm::run(case.as_str(), &config)?;
        }
    }

    Ok(())
}
