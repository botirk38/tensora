use std::path::Path;
use std::process;

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

    /// Number of partitions
    #[arg(short, long)]
    partitions: usize,

    /// I/O backend to use for conversion
    #[arg(short, long, default_value = "sync")]
    engine: Engine,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum Engine {
    Sync,
    Tokio,
    #[cfg(target_os = "linux")]
    IoUring,
}

fn main() {
    use clap::Parser;
    let args = Args::parse();

    let partition_count = args.partitions;

    if partition_count == 0 {
        eprintln!("Error: partition_count must be greater than 0");
        process::exit(1);
    }

    println!("Converting SafeTensors to ServerlessLLM format:");
    println!("  Input: {}", args.input_dir);
    println!("  Output: {}", args.output_dir);
    println!("  Partitions: {}", partition_count);
    println!("  Engine: {:?}", args.engine);

    // Create the converter entity with the requested engine preference
    let converter = tensora::SafeTensorsToServerlessLLM::new(
        Path::new(&args.input_dir),
        Path::new(&args.output_dir),
        partition_count,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error: Failed to create converter: {e}");
        std::process::exit(1);
    });

    let engine_preference = match args.engine {
        Engine::Sync => tensora::ConversionEnginePreference::Sync,
        Engine::Tokio => tensora::ConversionEnginePreference::Tokio,
        #[cfg(target_os = "linux")]
        Engine::IoUring => tensora::ConversionEnginePreference::IoUring,
    };

    let converter = converter.with_engine(engine_preference);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        match args.engine {
            Engine::Sync => converter.convert_sync(),
            _ => converter.convert_async().await,
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
        assert!(help.contains("--engine"));
    }

    #[test]
    fn engine_value_variants_include_core_choices() {
        use clap::ValueEnum;

        let mut names = Engine::value_variants()
            .iter()
            .filter_map(|variant| variant.to_possible_value())
            .map(|value| value.get_name().to_string())
            .collect::<Vec<_>>();
        names.sort();

        assert!(names.contains(&"sync".to_string()));
        assert!(names.contains(&"tokio".to_string()));
    }
}
