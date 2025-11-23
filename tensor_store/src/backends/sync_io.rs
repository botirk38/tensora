//! Synchronous blocking I/O using std::fs

use super::{IoResult, buffer_slice::BufferSlice, get_buffer_pool};
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;
use std::thread;

/// Ceiling division: (a + b - 1) / b
#[inline]
const fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Load entire file with blocking I/O
pub fn load(path: impl AsRef<Path>) -> IoResult<Vec<u8>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();

    let len = usize::try_from(file_len)
        .map_err(|_foo| Error::new(ErrorKind::InvalidInput, "file too large"))?;

    let mut buf = get_buffer_pool().get(len);
    file.read_exact(&mut buf[..])?;
    Ok(buf.into_inner())
}

/// Load tensor data in parallel chunks using synchronous I/O with zero-copy
#[inline]
pub fn load_parallel<P: AsRef<Path>>(path: P, chunks: usize) -> IoResult<Vec<u8>> {
    if chunks == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "chunks must be greater than 0",
        ));
    }

    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let metadata = file.metadata()?;
    let file_size = usize::try_from(metadata.len())
        .map_err(|_e| std::io::Error::new(std::io::ErrorKind::InvalidInput, "file too large"))?;

    let chunk_size = div_ceil(file_size, chunks);

    // Pre-allocate final pooled buffer - this is the ONLY allocation
    // (zero-copy: tasks write directly into non-overlapping buffer slices)
    let mut final_buf = get_buffer_pool().get(file_size);

    // SAFETY: We split final_buf into non-overlapping mutable slices.
    // Each slice is passed to exactly one task via BufferSlice.
    // The buffer remains valid until all tasks complete (we join all handles).
    // No other code accesses final_buf until all tasks are done.
    let mut handles = Vec::with_capacity(chunks);

    for i in 0..chunks {
        let start = i.checked_mul(chunk_size).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunk calculation overflow",
            )
        })?;
        let end = std::cmp::min(
            start.checked_add(chunk_size).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "chunk calculation overflow",
                )
            })?,
            file_size,
        );
        let actual_chunk_size = end.checked_sub(start).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "chunk calculation underflow",
            )
        })?;

        if actual_chunk_size == 0 {
            break;
        }

        // Create non-overlapping mutable slice
        let chunk_slice = final_buf
            .as_mut_slice()
            .get_mut(start..end)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid chunk range")
            })?;

        // SAFETY: This slice is unique to this task and won't be accessed elsewhere
        let mut buffer_slice = unsafe { BufferSlice::from_slice(chunk_slice) };

        let mut file_clone = file.try_clone()?;
        let handle = thread::spawn(move || {
            // Seek to the correct position
            file_clone.seek(SeekFrom::Start(u64::try_from(start).map_err(|_e| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek offset too large")
            })?))?;

            // SAFETY: We're the only task with access to this BufferSlice
            let slice = unsafe { buffer_slice.as_mut_slice() };

            // Read directly into the final buffer slice (ZERO COPY!)
            file_clone.read_exact(slice)?;

            IoResult::Ok(())
        });

        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle
            .join()
            .map_err(|_e| std::io::Error::other("thread panicked"))??;
    }

    // All data is now in final_buf, set correct length
    final_buf.truncate(file_size);
    Ok(final_buf.into_inner())
}

/// Load byte range with blocking I/O
pub fn load_range(path: impl AsRef<Path>, offset: u64, len: usize) -> IoResult<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;

    let mut buf = get_buffer_pool().get(len);
    file.read_exact(&mut buf[..])?;
    Ok(buf.into_inner())
}
