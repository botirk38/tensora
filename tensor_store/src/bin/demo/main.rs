use clap::{Parser, Subcommand, ValueEnum};

mod config;
mod safetensors;
mod serverlessllm;

use config::DemoConfig;

#[derive(Parser)]
#[command(name = "tensor_store demo")]
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

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,
    },
    /// Demonstrate ServerlessLLM loader
    Serverlessllm {
        /// Scenario to run
        #[arg(value_enum)]
        scenario: ServerlessLLMScenario,

        /// Fixture name (e.g., qwen2-0.5b, mistral-7b)
        #[arg(short, long)]
        fixture: Option<String>,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SafeTensorsScenario {
    /// Async loading with io_uring (Linux) or tokio (other platforms)
    Async,
    /// Synchronous loading
    Sync,
    /// Memory-mapped lazy loading
    Mmap,
    /// Parallel multi-core loading
    Parallel,
    /// Detailed tensor metadata exploration
    Metadata,
    /// Run all scenarios sequentially
    All,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum ServerlessLLMScenario {
    /// Async loading with partition information
    Async,
    /// Synchronous loading
    Sync,
    /// Memory-mapped lazy loading
    Mmap,
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
            Self::Parallel => "parallel",
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
            Self::Metadata => "metadata",
            Self::All => "all",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Safetensors { scenario, fixture } => {
            let config = DemoConfig { fixture };
            safetensors::run(scenario.as_str(), &config)?;
        }
        Commands::Serverlessllm { scenario, fixture } => {
            let config = DemoConfig { fixture };
            serverlessllm::run(scenario.as_str(), &config)?;
        }
    }

    Ok(())
}
