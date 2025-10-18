#[cfg(target_os = "linux")]
fn main() -> std::io::Result<()> {
    use tensor_store::FixedBufPool;

    tokio_uring::start(async {
        let path = "test_model.safetensors";

        println!("Creating fixed buffer pool...");
        let bufs: Vec<Vec<u8>> = (0..4).map(|_| vec![0u8; 64 * 1024 * 1024]).collect();
        let buf_pool = FixedBufPool::new(bufs);
        buf_pool.register()?;

        println!("Loading safetensors file with fixed buffers: {}", path);
        let data = tensor_store::load_safetensors_fixed(path, &buf_pool).await?;

        println!("Loaded {} bytes", data.len());

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("Number of tensors: {}", tensors.names().len());

        Ok(())
    })
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("Fixed buffer loading is only available on Linux with io_uring support");
}
