use clap::{Parser, Subcommand, ValueEnum};

mod config;
mod io_metrics;
mod safetensors;
mod serverlessllm;

use config::DemoConfig;

#[derive(Parser)]
#[command(name = "tensora demo")]
#[command(about = "Showcase SafeTensors and ServerlessLLM loaders", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Demonstrate SafeTensors loader
    Safetensors {
        /// Scenario to run
        #[arg(value_enum)]
        scenario: SafeTensorsScenario,

        /// Hugging Face model id
        #[arg(long)]
        model_id: String,
    },
    /// Demonstrate ServerlessLLM loader
    Serverlessllm {
        /// Scenario to run
        #[arg(value_enum)]
        scenario: ServerlessLLMScenario,

        #[arg(long)]
        model_id: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsScenario {
    /// Async sequential loading (io_uring on Linux, tokio elsewhere)
    Async,
    /// Synchronous loading
    Sync,
    /// Memory-mapped lazy loading
    Mmap,
    /// Sync parallel multi-core loading (blocking I/O, multiple threads)
    ParallelSync,
    /// Detailed tensor metadata exploration
    Metadata,
    /// Run all scenarios sequentially
    All,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMScenario {
    /// Async sequential loading (io_uring on Linux, tokio elsewhere)
    Async,
    /// Synchronous loading
    Sync,
    /// Memory-mapped lazy loading
    Mmap,
    /// Sync parallel multi-core loading (blocking I/O, multiple threads)
    ParallelSync,
    /// Index structure and partition statistics
    Metadata,
    /// Run all scenarios sequentially
    All,
}

impl SafeTensorsScenario {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Async => "async",
            Self::Sync => "sync",
            Self::Mmap => "mmap",
            Self::ParallelSync => "parallel-sync",
            Self::Metadata => "metadata",
            Self::All => "all",
        }
    }
}

impl ServerlessLLMScenario {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Async => "async",
            Self::Sync => "sync",
            Self::Mmap => "mmap",
            Self::ParallelSync => "parallel-sync",
            Self::Metadata => "metadata",
            Self::All => "all",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Safetensors { scenario, model_id } => {
            let config = DemoConfig { model_id };
            safetensors::run(scenario.as_str(), &config)?;
        }
        Commands::Serverlessllm { scenario, model_id } => {
            let config = DemoConfig { model_id };
            serverlessllm::run(scenario.as_str(), &config)?;
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
        assert!(help.contains("Showcase SafeTensors and ServerlessLLM loaders"));
        assert!(help.contains("safetensors"));
        assert!(help.contains("serverlessllm"));
    }

    #[test]
    fn scenario_string_mappings_are_stable() {
        assert_eq!(SafeTensorsScenario::ParallelSync.as_str(), "parallel-sync");
        assert_eq!(SafeTensorsScenario::Metadata.as_str(), "metadata");
        assert_eq!(ServerlessLLMScenario::ParallelSync.as_str(), "parallel-sync");
        assert_eq!(ServerlessLLMScenario::All.as_str(), "all");
    }
}
