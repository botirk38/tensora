use std::env;

mod config;
mod safetensors;
mod serverlessllm;

use config::{ProfileConfig, ProfileError};

#[derive(Debug, Clone, Copy)]
enum Suite {
    Safetensors,
    ServerlessLlm,
}

#[derive(Debug)]
struct CliArgs {
    suite: Suite,
    case: String,
    fixture: Option<String>,
    iterations: usize,
    list: bool,
    help: bool,
}

fn parse_args() -> Result<CliArgs, ProfileError> {
    let mut suite: Option<Suite> = None;
    let mut case: Option<String> = None;
    let mut fixture: Option<String> = None;
    let mut iterations: usize = 1;
    let mut list = false;
    let mut help = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--suite" => {
                let value = args.next().ok_or_else(|| {
                    ProfileError::new("Missing value after --suite (safetensors|serverlessllm)")
                })?;
                suite = match value.as_str() {
                    "safetensors" => Some(Suite::Safetensors),
                    "serverlessllm" => Some(Suite::ServerlessLlm),
                    other => {
                        return Err(ProfileError::new(format!(
                            "Unknown suite '{}'. Use safetensors or serverlessllm",
                            other
                        )));
                    }
                };
            }
            "--case" => {
                case = Some(args.next().ok_or_else(|| {
                    ProfileError::new("Missing value after --case (see --list for options)")
                })?);
            }
            "--fixture" => {
                fixture = Some(
                    args.next()
                        .ok_or_else(|| ProfileError::new("Missing value after --fixture"))?,
                );
            }
            "--iterations" => {
                let value = args
                    .next()
                    .ok_or_else(|| ProfileError::new("Missing value after --iterations"))?;
                iterations = value.parse::<usize>().map_err(|err| {
                    ProfileError::new(format!(
                        "--iterations expects a positive integer (got '{value}': {err})"
                    ))
                })?;
            }
            "--list" => {
                list = true;
            }
            "--help" | "-h" => {
                help = true;
            }
            other => return Err(ProfileError::new(format!("Unknown argument '{}'", other))),
        }
    }

    let selected_suite = suite.ok_or_else(|| {
        ProfileError::new("Missing --suite. Use safetensors or serverlessllm (try --list)")
    })?;

    let selected_case = if list || help {
        // When listing or printing help, we don't need a case value.
        case.unwrap_or_default()
    } else {
        case.ok_or_else(|| {
            ProfileError::new("Missing --case. Use --list after selecting a suite to see options")
        })?
    };

    Ok(CliArgs {
        suite: selected_suite,
        case: selected_case,
        fixture,
        iterations,
        list,
        help,
    })
}

fn print_usage() {
    eprintln!(
        "\
Profiling harness (no Criterion).

Usage:
  cargo run --release --bin profile -- --suite <safetensors|serverlessllm> --case <name> [--fixture <name>] [--iterations N]

Options:
  --suite        Target suite to profile (safetensors or serverlessllm)
  --case         Scenario to run (use --list with --suite to see options)
  --fixture      Fixture directory under ./fixtures (default: all fixtures)
  --iterations   Number of iterations to run (default: 1)
  --list         List available cases for the chosen suite and exit
  --help, -h     Show this message

Examples:
  cargo flamegraph --bin profile -- --suite safetensors --case io-uring-load --fixture qwen2-0.5b
  cargo flamegraph --bin profile -- --suite serverlessllm --case async-load --iterations 3
"
    );
}

fn list_cases(suite: Suite) {
    match suite {
        Suite::Safetensors => {
            println!("safetensors cases:");
            for case in safetensors::available_cases() {
                println!("  - {case}");
            }
        }
        Suite::ServerlessLlm => {
            println!("serverlessllm cases:");
            for case in serverlessllm::available_cases() {
                println!("  - {case}");
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    if args.help {
        print_usage();
        return Ok(());
    }

    if args.list {
        list_cases(args.suite);
        return Ok(());
    }

    let config = ProfileConfig {
        iterations: args.iterations,
        fixture: args.fixture,
    };

    match args.suite {
        Suite::Safetensors => safetensors::run(&args.case, &config)?,
        Suite::ServerlessLlm => serverlessllm::run(&args.case, &config)?,
    }

    Ok(())
}
