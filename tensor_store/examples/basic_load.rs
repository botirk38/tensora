#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    tokio_uring::start(async {
        let path = "test_model.safetensors";

        println!("Loading safetensors file: {}", path);
        let data = tensor_store::load_safetensors(path).await?;

        println!("Loaded {} bytes", data.len());

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("Number of tensors: {}", tensors.names().len());
        for name in tensors.names() {
            println!("  - {}", name);
        }

        Ok(())
    })
}

#[cfg(not(target_os = "linux"))]
#[tokio::main]
async fn main() -> std::io::Result<()> {
    let path = "test_model.safetensors";

    println!("Loading safetensors file: {}", path);
    let data = tensor_store::load_safetensors(path).await?;

    println!("Loaded {} bytes", data.len());

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    println!("Number of tensors: {}", tensors.names().len());
    for name in tensors.names() {
        println!("  - {}", name);
    }

    Ok(())
}
