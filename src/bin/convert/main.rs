use std::env;
use std::process;

#[cfg(target_os = "linux")]
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "Usage: {} <input.safetensors> <output_dir> [partition_count]",
            args[0]
        );
        eprintln!("  partition_count defaults to number of CPU cores");
        process::exit(1);
    }

    let input_path = args.get(1).unwrap();
    let output_dir = args.get(2).unwrap();
    let partition_count = if args.len() == 4 {
        args.get(3).unwrap().parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Error: partition_count must be a positive integer");
            process::exit(1);
        })
    } else {
        num_cpus::get()
    };

    if partition_count == 0 {
        eprintln!("Error: partition_count must be greater than 0");
        process::exit(1);
    }

    println!("Converting SafeTensors to ServerlessLLM format:");
    println!("  Input: {}", input_path);
    println!("  Output: {}", output_dir);
    println!("  Partitions: {}", partition_count);

    tokio_uring::start(async {
        match tensor_store::convert_safetensors_to_serverlessllm(
            input_path,
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

#[cfg(not(target_os = "linux"))]
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "Usage: {} <input.safetensors> <output_dir> [partition_count]",
            args[0]
        );
        eprintln!("  partition_count defaults to number of CPU cores");
        process::exit(1);
    }

    let input_path = args.get(1).unwrap();
    let output_dir = args.get(2).unwrap();
    let partition_count = if args.len() == 4 {
        args.get(3).unwrap().parse::<usize>().unwrap_or_else(|_| {
            eprintln!("Error: partition_count must be a positive integer");
            process::exit(1);
        })
    } else {
        num_cpus::get()
    };

    if partition_count == 0 {
        eprintln!("Error: partition_count must be greater than 0");
        process::exit(1);
    }

    println!("Converting SafeTensors to ServerlessLLM format:");
    println!("  Input: {}", input_path);
    println!("  Output: {}", output_dir);
    println!("  Partitions: {}", partition_count);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        match tensor_store::convert_safetensors_to_serverlessllm(
            input_path,
            output_dir,
            partition_count,
        )
        .await
        {
            Ok(_) => {
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
