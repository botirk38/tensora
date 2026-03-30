use std::env;
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
    tensor_store::recommended_partition_count(total)
}

#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "Usage: {} <input_dir> <output_dir> [partition_count]",
            args[0]
        );
        eprintln!("  input_dir must contain one or more .safetensors shards");
        process::exit(1);
    }

    let input_dir = args.get(1).unwrap();
    let output_dir = args.get(2).unwrap();
    let partition_count = if args.len() == 4 {
        args.get(3).unwrap().parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Error: partition_count must be a positive integer");
            process::exit(1);
        })
    } else {
        default_partition_count(input_dir)
    };

    if partition_count == 0 {
        eprintln!("Error: partition_count must be greater than 0");
        process::exit(1);
    }

    println!("Converting SafeTensors to ServerlessLLM format:");
    println!("  Input: {}", input_dir);
    println!("  Output: {}", output_dir);
    println!("  Partitions: {}", partition_count);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        match tensor_store::convert_safetensors_to_serverlessllm(
            input_dir,
            output_dir,
            partition_count,
        )
        .await
        {
            Ok(()) => {
                println!("✓ Conversion completed successfully!");
                println!("  Index: {}/tensor_index.json", output_dir);
                println!(
                    "  Data files: {}/tensor.data_0 ... {}/tensor.data_{}",
                    output_dir,
                    output_dir,
                    partition_count - 1
                );
            }
            Err(e) => {
                eprintln!("Error during conversion: {}", e);
                process::exit(1);
            }
        }
    });
}
