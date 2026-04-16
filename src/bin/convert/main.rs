use std::process;

fn default_partition_count(input_dir: &str) -> usize {
    let mut total = 0u64;
    let entries = std::fs::read_dir(input_dir).unwrap_or_else(|e| {
        eprintln!("Error reading directory {input_dir}: {e}");
        process::exit(1);
    });
    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| {
                eprintln!("Error reading directory entry in {input_dir}: {e}");
                process::exit(1);
            })
            .path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".safetensors"))
        {
            total += path
                .metadata()
                .unwrap_or_else(|e| {
                    eprintln!("Error reading metadata for {}: {e}", path.display());
                    process::exit(1);
                })
                .len();
        }
    }
    if total == 0 {
        eprintln!("Error: no .safetensors files found in {input_dir}");
        process::exit(1);
    }
    tensora::recommended_partition_count(total)
}

#[derive(clap::Parser, Debug)]
#[command(
    name = "convert",
    about = "Convert SafeTensors to ServerlessLLM format"
)]
struct Args {
    /// Input directory containing .safetensors shards
    input_dir: String,

    /// Output directory for ServerlessLLM format
    output_dir: String,

    /// Number of partitions (default: auto from model size)
    #[arg(short, long)]
    partitions: Option<usize>,

    /// Backend to use for conversion (default: adaptive)
    #[arg(short, long, default_value = "default")]
    backend: Backend,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Backend {
    Default,
    Sync,
    Async,
    #[cfg(target_os = "linux")]
    IoUring,
}

fn main() {
    use clap::Parser;
    let args = Args::parse();

    let partition_count = args
        .partitions
        .unwrap_or_else(|| default_partition_count(&args.input_dir));

    if partition_count == 0 {
        eprintln!("Error: partition_count must be greater than 0");
        process::exit(1);
    }

    println!("Converting SafeTensors to ServerlessLLM format:");
    println!("  Input: {}", args.input_dir);
    println!("  Output: {}", args.output_dir);
    println!("  Partitions: {}", partition_count);
    println!("  Backend: {:?}", args.backend);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        match args.backend {
            Backend::Default => {
                tensora::convert_safetensors_to_serverlessllm(
                    &args.input_dir,
                    &args.output_dir,
                    partition_count,
                )
                .await
            }
            Backend::Sync => tensora::convert_safetensors_to_serverlessllm_sync(
                &args.input_dir,
                &args.output_dir,
                partition_count,
            ),
            Backend::Async => {
                tensora::convert_safetensors_to_serverlessllm_async(
                    &args.input_dir,
                    &args.output_dir,
                    partition_count,
                )
                .await
            }
            #[cfg(target_os = "linux")]
            Backend::IoUring => tensora::convert_safetensors_to_serverlessllm_io_uring(
                &args.input_dir,
                &args.output_dir,
                partition_count,
            ),
        }
    });

    match result {
        Ok(()) => {
            println!("Conversion completed successfully!");
            println!("  Index: {}/tensor_index.json", args.output_dir);
            println!(
                "  Data files: {}/tensor.data_0 ... {}/tensor.data_{}",
                args.output_dir,
                args.output_dir,
                partition_count - 1
            );
        }
        Err(e) => {
            eprintln!("Error during conversion: {}", e);
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn clap_help_contains_summary() {
        let mut cmd = Args::command();
        let help = cmd.render_long_help().to_string();
        assert!(help.contains("Convert SafeTensors to ServerlessLLM format"));
        assert!(help.contains("--partitions"));
        assert!(help.contains("--backend"));
    }

    #[test]
    fn backend_value_variants_include_core_choices() {
        use clap::ValueEnum;

        let mut names = Backend::value_variants()
            .iter()
            .filter_map(|variant| variant.to_possible_value())
            .map(|value| value.get_name().to_string())
            .collect::<Vec<_>>();
        names.sort();

        assert!(names.contains(&"default".to_string()));
        assert!(names.contains(&"sync".to_string()));
        assert!(names.contains(&"async".to_string()));
    }
}
