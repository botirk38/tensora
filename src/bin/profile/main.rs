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
    /// Print storage-engine availability in the current process/environment
    Capabilities {
        /// Output format
        #[arg(long, value_enum, default_value_t = CapabilityFormat::Text)]
        format: CapabilityFormat,
    },

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
enum CapabilityFormat {
    Text,
    Shell,
    Json,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Tokio asynchronous load
    Tokio,
    /// Memory-mapped open
    Mmap,
    /// Upstream safetensors crate baseline (std::fs::read + deserialize)
    HfNative,
    /// Explicit io_uring storage engine (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMCase {
    /// Heuristic default load
    Default,
    /// Synchronous load
    Sync,
    /// Tokio asynchronous load
    Tokio,
    /// Memory-mapped open
    Mmap,
    /// Explicit io_uring storage engine (Linux only)
    #[cfg(target_os = "linux")]
    IoUring,
}

impl SafeTensorsCase {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Sync => "sync",
            Self::Tokio => "tokio",
            Self::Mmap => "mmap",
            Self::HfNative => "hf-native",
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
            Self::Tokio => "tokio",
            Self::Mmap => "mmap",
            #[cfg(target_os = "linux")]
            Self::IoUring => "io-uring",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Capabilities { format } => print_capabilities(format)?,
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

fn print_capabilities(format: CapabilityFormat) -> Result<(), Box<dyn std::error::Error>> {
    use tensora::storage::availability::{StorageAvailability, StorageCapabilities};
    let capabilities = StorageCapabilities::probe();
    match format {
        CapabilityFormat::Text => {
            for (kind, availability) in capabilities.iter() {
                println!("{kind:<10} {availability}");
            }
        }
        CapabilityFormat::Shell => {
            for (kind, availability) in capabilities.iter() {
                let key = kind.as_str().replace('-', "_").to_ascii_uppercase();
                println!(
                    "TENSORA_ENGINE_{key}_AVAILABLE={}",
                    availability.is_available()
                );
            }
        }
        CapabilityFormat::Json => {
            let mut entries = serde_json::Map::new();
            for (kind, availability) in capabilities.iter() {
                let value = match availability {
                    StorageAvailability::Available => serde_json::json!({
                        "available": true,
                    }),
                    StorageAvailability::Unavailable { reason, details } => {
                        serde_json::json!({
                            "available": false,
                            "reason": reason.code(),
                            "details": details,
                        })
                    }
                };
                entries.insert(kind.as_str().to_owned(), value);
            }
            println!("{}", serde_json::Value::Object(entries));
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
        assert!(help.contains("capabilities"));
        assert!(help.contains("safetensors"));
        assert!(help.contains("serverlessllm"));
    }

    #[test]
    fn case_string_mappings_are_stable() {
        assert_eq!(SafeTensorsCase::Default.as_str(), "default");
        assert_eq!(SafeTensorsCase::Sync.as_str(), "sync");
        assert_eq!(ServerlessLLMCase::Tokio.as_str(), "tokio");
        assert_eq!(ServerlessLLMCase::Mmap.as_str(), "mmap");
    }
}
