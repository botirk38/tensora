#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    tokio_uring::start(async {
        let path = "test_model.safetensors";

        println!("Loading safetensors file with parallel I/O: {}", path);
        let data = tensor_store::load_safetensors_parallel(path).await?;

        println!("Loaded {} bytes", data.len());

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("Number of tensors: {}", tensors.names().len());

        Ok(())
    })
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("Parallel loading is only available on Linux with io_uring support");
}
