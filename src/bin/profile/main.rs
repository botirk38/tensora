use clap::{Parser, Subcommand, ValueEnum};

mod config;
mod evict;
mod safetensors;
mod serverlessllm;
mod stats;

use config::ProfileConfig;

#[derive(Parser)]
#[command(name = "profile")]
#[command(about = "Profiling harness for tensora (no Criterion)", long_about = None)]
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

        /// Hugging Face model id (weights are fetched into the Hub cache if needed)
        #[arg(long)]
        model_id: String,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,

        /// Evict kernel page cache between iterations
        #[arg(long, default_value_t = false)]
        evict_page_cache: bool,
    },
    /// Profile ServerlessLLM loader
    Serverlessllm {
        /// Profiling case to run
        #[arg(value_enum)]
        case: ServerlessLLMCase,

        #[arg(long)]
        model_id: String,

        /// Number of iterations to run (default: 1)
        #[arg(short, long, default_value_t = 1)]
        iterations: usize,

        /// Evict kernel page cache between iterations
        #[arg(long, default_value_t = false)]
        evict_page_cache: bool,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Asynchronous load
    Async,
    /// Memory-mapped open
    Mmap,
    /// Explicit io_uring backend (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Asynchronous load
    Async,
    /// Memory-mapped open
    Mmap,
    /// Explicit io_uring backend (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

impl SafeTensorsCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Sync => "sync",
            Self::Async => "async",
            Self::Mmap => "mmap",
            #[cfg(target_os = "linux")]
            Self::IoUring => "io-uring",
        }
    }
}

impl ServerlessLLMCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Sync => "sync",
            Self::Async => "async",
            Self::Mmap => "mmap",
            #[cfg(target_os = "linux")]
            Self::IoUring => "io-uring",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Safetensors {
            case,
            model_id,
            iterations,
            evict_page_cache,
        } => {
            let config = ProfileConfig {
                iterations,
                model_id,
                evict_page_cache,
            };
            safetensors::run(case.as_str(), &config)?;
        }
        Commands::Serverlessllm {
            case,
            model_id,
            iterations,
            evict_page_cache,
        } => {
            let config = ProfileConfig {
                iterations,
                model_id,
                evict_page_cache,
            };
            serverlessllm::run(case.as_str(), &config)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_help_contains_summary() {
        let mut cmd = Cli::command();
        let help = cmd.render_long_help().to_string();
        assert!(help.contains("Profiling harness for tensora"));
        assert!(help.contains("safetensors"));
        assert!(help.contains("serverlessllm"));
    }

    #[test]
    fn case_string_mappings_are_stable() {
        assert_eq!(SafeTensorsCase::Default.as_str(), "default");
        assert_eq!(SafeTensorsCase::Sync.as_str(), "sync");
        assert_eq!(ServerlessLLMCase::Async.as_str(), "async");
        assert_eq!(ServerlessLLMCase::Mmap.as_str(), "mmap");
    }
}
