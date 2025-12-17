//! `ServerlessLLM` format reader.
//!
//! This module provides functionality to parse `ServerlessLLM` tensor index files,
//! load tensor data from partition files, and validate partition integrity.
//!
//! # Format Structure
//!
//! ```text
//! tensor_index.json:
//! {
//!   "tensor_name": [offset, size, [shape...], [stride...], "dtype", partition_id],
//!   ...
//! }
//!
//! tensor.data_0: Binary tensor data (partition 0)
//! tensor.data_1: Binary tensor data (partition 1)
//! ...
//! ```
//!
//! # Usage Examples
//!
//! ## High-Level API (Recommended)
//!
//! ```rust,ignore
//! use tensor_store::serverlessllm;
//!
//! // Load entire model with eager loading (async)
//! let model = serverlessllm::load("model_dir/").await?;
//!
//! // Or synchronously
//! let model = serverlessllm::load_sync("model_dir/")?;
//!
//! // Lazy mmap loading (sync)
//! let model_mmap = serverlessllm::load_mmap("model_dir/")?;
//! println!("Loaded {} tensors", model.len());
//!
//! // Access tensors by name
//! let weights = model.tensor("layer.0.weight").unwrap();
//! println!("Shape: {:?}, Size: {} bytes", weights.shape(), weights.size());
//!
//! // Iterate over all tensors
//! for (name, tensor) in &model {
//!     println!("{}: {} bytes", name, tensor.data().len());
//! }
//! ```
//!
//! ## Low-Level API (Advanced)
//!
//! For fine-grained control, use the index-based API:
//!
//! ```rust,ignore
//! // Parse index file
//! // Async API
//! let index = serverlessllm::parse_index("model/tensor_index.json").await?;
//!
//! // Sync API
//! let index = serverlessllm::parse_index_sync("model/tensor_index.json")?;
//!
//! // Load individual tensors
//! let weight = index.load_tensor("model/tensor.data", "layer.0.weight").await?;
//!
//! // Or load multiple tensors concurrently
//! let tensors = index.load_tensors_batch("model/tensor.data", &["layer.0.weight", "layer.0.bias"]).await?;
//! ```
//!
use crate::backends;
use crate::serverlessllm::types::TensorEntry;
use crate::types::error::{ReaderError, ReaderResult};
use crate::types::traits::{AsyncReader, SyncReader, TensorMetadata};
use futures::future;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
#[cfg(target_os = "linux")]
use tokio_uring::fs::statx;

/// Parsed `ServerlessLLM` index
#[derive(Debug, Default)]
pub struct ServerlessLLMIndex {
    /// All tensors in the index
    tensors: HashMap<String, TensorEntry>,
    /// Cache partition metadata: partition_id -> file_size
    partition_cache: Mutex<HashMap<usize, u64>>,
}

impl Clone for ServerlessLLMIndex {
    fn clone(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
            partition_cache: Mutex::new(HashMap::new()), // Fresh cache for clone
        }
    }
}

impl ServerlessLLMIndex {
    /// Creates a new empty index.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            partition_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a reference to the tensors map.
    #[inline]
    #[must_use]
    pub const fn tensors(&self) -> &HashMap<String, TensorEntry> {
        &self.tensors
    }

    /// Gets a tensor entry by name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    /// Returns the number of tensors tracked by this index.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true when the index has no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns tensor names without requiring the `TensorMetadata` trait in scope.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Returns an immutable view of the underlying tensor map.
    #[inline]
    #[must_use]
    pub const fn tensors_map(&self) -> &HashMap<String, TensorEntry> {
        &self.tensors
    }

    /// Returns an iterator over tensor names and entries.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorEntry)> {
        self.tensors.iter()
    }

    /// Loads tensor data from a partition file asynchronously.
    ///
    /// Automatically validates that the partition file exists and has sufficient size
    /// before attempting to load the data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_name` - Name of the tensor to load
    ///
    /// # Returns
    ///
    /// The raw tensor data as bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor name is not found in the index
    /// - The partition file doesn't exist
    /// - The partition file is too small
    /// - I/O errors occur during reading
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let index = ServerlessLLMIndex::load("model/tensor_index.json").await?;
    /// let data = index.load_tensor("model/tensor.data", "layer.0.weight").await?;
    /// ```
    #[inline]
    pub async fn load_tensor(
        &self,
        base_path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> ReaderResult<Vec<u8>> {
        let entry = self.tensors.get(tensor_name).ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor '{tensor_name}' not found in index"))
        })?;

        let base_path_str = base_path.as_ref().to_string_lossy();
        let partition_path = format!("{}_{}", base_path_str, entry.partition_id);

        // Validate partition file before loading
        self.validate_single_partition_async(&partition_path, entry)
            .await?;

        let data = backends::load_range(
            &partition_path,
            entry.offset,
            usize::try_from(entry.size)
                .map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?,
        )
        .await
        .map_err(ReaderError::from)?;

        // Convert pooled buffer to Vec<u8> for API compatibility
        Ok(data)
    }

    /// Loads tensor data from a partition file synchronously.
    ///
    /// Automatically validates that the partition file exists and has sufficient size
    /// before attempting to load the data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_name` - Name of the tensor to load
    ///
    /// # Returns
    ///
    /// The raw tensor data as bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor name is not found in the index
    /// - The partition file doesn't exist
    /// - The partition file is too small
    /// - I/O errors occur during reading
    #[inline]
    pub fn load_tensor_sync(
        &self,
        base_path: impl AsRef<Path>,
        tensor_name: &str,
    ) -> ReaderResult<Vec<u8>> {
        let entry = self.tensors.get(tensor_name).ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor '{tensor_name}' not found in index"))
        })?;

        let base_path_str = base_path.as_ref().to_string_lossy();
        let partition_path = format!("{}_{}", base_path_str, entry.partition_id);

        // Validate partition file before loading
        self.validate_single_partition(&partition_path, entry)?;

        let data = backends::sync::load_range(
            &partition_path,
            entry.offset,
            usize::try_from(entry.size)
                .map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?,
        )
        .map_err(ReaderError::from)?;

        // Convert pooled buffer to Vec<u8> for API compatibility
        Ok(data)
    }

    /// Loads multiple tensors concurrently using io_uring batching.
    ///
    /// This method loads tensors in parallel, leveraging io_uring's kernel-level
    /// batching for optimal performance when loading multiple tensors.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_names` - Names of tensors to load
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor names to their raw data.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor is not found or if I/O errors occur.
    /// All tensors are validated before loading begins.
    pub async fn load_tensors_batch(
        &self,
        base_path: impl AsRef<Path>,
        tensor_names: &[&str],
    ) -> ReaderResult<HashMap<String, Vec<u8>>> {
        let zero_copy_results = self
            .load_tensors_batch_zero_copy(base_path, tensor_names)
            .await?;

        // Convert results to owned Vec<u8> for API compatibility
        let result = zero_copy_results
            .into_iter()
            .map(|(name, (buf, offset, len))| {
                let data = if offset == 0 && len == buf.len() {
                    buf
                } else {
                    buf[offset..offset + len].to_vec()
                };
                (name, data)
            })
            .collect();

        Ok(result)
    }

    /// Loads multiple tensors concurrently using io_uring batching (zero-copy).
    ///
    /// This method loads tensors in parallel, leveraging io_uring's kernel-level
    /// batching for optimal performance when loading multiple tensors.
    /// Returns shared buffers with offset/length metadata for zero-copy access.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_names` - Names of tensors to load
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor names to (shared_buffer, offset, length) tuples.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor is not found or if I/O errors occur.
    /// All tensors are validated before loading begins.
    pub async fn load_tensors_batch_zero_copy(
        &self,
        base_path: impl AsRef<Path>,
        tensor_names: &[&str],
    ) -> ReaderResult<HashMap<String, (Vec<u8>, usize, usize)>> {
        use std::collections::HashMap as StdHashMap;

        // Validate all tensors exist first
        let mut entries = Vec::with_capacity(tensor_names.len());
        for &name in tensor_names {
            let entry = self.tensors.get(name).ok_or_else(|| {
                ReaderError::ServerlessLlm(format!("tensor '{name}' not found in index"))
            })?;
            entries.push((name.to_string(), entry));
        }

        // Group requests by partition file for true io_uring batching
        let mut partition_requests: StdHashMap<usize, Vec<(String, &TensorEntry)>> =
            StdHashMap::new();
        let base_path_str = base_path.as_ref().to_string_lossy();

        for (name, entry) in &entries {
            partition_requests
                .entry(entry.partition_id)
                .or_default()
                .push((name.clone(), entry));
        }

        // Build paths and stat partitions concurrently (non-blocking)
        let partition_paths: StdHashMap<usize, String> = partition_requests
            .keys()
            .map(|&partition_id| (partition_id, format!("{}_{}", base_path_str, partition_id)))
            .collect();

        let partition_sizes_vec = future::try_join_all(partition_paths.iter().map(
            |(partition_id, path)| async move {
                let size = stat_partition_size(path).await?;
                ReaderResult::Ok((*partition_id, size))
            },
        ))
        .await?;

        let partition_sizes: StdHashMap<usize, u64> = partition_sizes_vec.into_iter().collect();

        if let Ok(mut cache) = self.partition_cache.lock() {
            for (partition_id, size) in &partition_sizes {
                cache.insert(*partition_id, *size);
            }
        }

        let mut batch_requests = Vec::with_capacity(entries.len());
        let mut names = Vec::with_capacity(entries.len());

        for (name, entry) in &entries {
            let required_size = entry
                .offset
                .checked_add(entry.size)
                .ok_or_else(|| ReaderError::ServerlessLlm("offset + size overflow".to_owned()))?;

            let partition_path = partition_paths.get(&entry.partition_id).ok_or_else(|| {
                ReaderError::ServerlessLlm(format!(
                    "partition {} missing path resolution",
                    entry.partition_id
                ))
            })?;

            let actual_size = *partition_sizes.get(&entry.partition_id).ok_or_else(|| {
                ReaderError::ServerlessLlm(format!(
                    "partition file '{}' not found or inaccessible",
                    partition_path
                ))
            })?;

            if actual_size < required_size {
                return Err(ReaderError::ServerlessLlm(format!(
                    "partition file '{}' is too small: has {actual_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {}",
                    partition_path, entry.offset, entry.size
                )));
            }

            let len = usize::try_from(entry.size)
                .map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?;

            batch_requests.push((partition_path.as_str(), entry.offset, len));
            names.push(name.clone());
        }

        let batch_results = backends::load_batch(&batch_requests)
            .await
            .map_err(ReaderError::from)?;

        Ok(names.into_iter().zip(batch_results).collect())
    }

    /// Loads multiple tensors concurrently using io_uring batching (sync version).
    ///
    /// This method loads tensors in parallel using thread-based batching for optimal
    /// performance when loading multiple tensors synchronously.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_names` - Names of tensors to load
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor names to their raw data.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor is not found or if I/O errors occur.
    /// All tensors are validated before loading begins.
    pub fn load_tensors_batch_sync(
        &self,
        base_path: impl AsRef<Path>,
        tensor_names: &[&str],
    ) -> ReaderResult<HashMap<String, Vec<u8>>> {
        use std::collections::HashMap as StdHashMap;

        // Validate all tensors exist first
        let mut entries = Vec::with_capacity(tensor_names.len());
        for &name in tensor_names {
            let entry = self.tensors.get(name).ok_or_else(|| {
                ReaderError::ServerlessLlm(format!("tensor '{name}' not found in index"))
            })?;
            entries.push((name.to_string(), entry));
        }

        // Build partition paths
        let base_path_str = base_path.as_ref().to_string_lossy();
        let mut partition_paths: StdHashMap<usize, String> = StdHashMap::new();

        for (_, entry) in &entries {
            partition_paths
                .entry(entry.partition_id)
                .or_insert_with(|| format!("{}_{}", base_path_str, entry.partition_id));
        }

        // Collect unique partition validations needed
        let mut partition_validations: StdHashMap<usize, (String, u64)> = StdHashMap::new();
        for (_, entry) in &entries {
            let required_size = entry
                .offset
                .checked_add(entry.size)
                .ok_or_else(|| ReaderError::ServerlessLlm("offset + size overflow".to_owned()))?;

            partition_validations
                .entry(entry.partition_id)
                .and_modify(|(_, max_size)| *max_size = (*max_size).max(required_size))
                .or_insert_with(|| {
                    (
                        partition_paths.get(&entry.partition_id).unwrap().clone(),
                        required_size,
                    )
                });
        }

        // Parallel validation of all unique partitions
        let validation_results: Vec<Result<(usize, u64), ReaderError>> = partition_validations
            .par_iter()
            .map(|(&partition_id, (path, required_size))| {
                let metadata = std::fs::metadata(path).map_err(|e| {
                    ReaderError::ServerlessLlm(format!(
                        "partition file '{}' not found or inaccessible: {e}",
                        path
                    ))
                })?;
                let actual_size = metadata.len();
                if actual_size < *required_size {
                    return Err(ReaderError::ServerlessLlm(format!(
                        "partition file '{}' is too small: has {actual_size} bytes, needs at least {required_size} bytes",
                        path
                    )));
                }
                Ok((partition_id, actual_size))
            })
            .collect();

        // Check for any validation errors and cache results
        for result in validation_results {
            let (partition_id, actual_size) = result?;
            if let Ok(mut cache) = self.partition_cache.lock() {
                cache.insert(partition_id, actual_size);
            }
        }

        // Build batch requests: (path, offset, len)
        let mut batch_requests: Vec<(String, u64, usize)> = Vec::with_capacity(entries.len());
        let mut names = Vec::with_capacity(entries.len());

        for (name, entry) in &entries {
            let partition_path = partition_paths.get(&entry.partition_id).unwrap().clone();
            let len = usize::try_from(entry.size)
                .map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?;
            batch_requests.push((partition_path, entry.offset, len));
            names.push(name.clone());
        }

        // Load all tensors in parallel using batch API
        let batch_results =
            backends::sync::load_range_batch(&batch_requests).map_err(ReaderError::from)?;

        // Convert results to HashMap
        let mut result = HashMap::with_capacity(names.len());
        for (name, (buf, offset, len)) in names.into_iter().zip(batch_results) {
            let data = if offset == 0 && len == buf.len() {
                buf
            } else {
                buf[offset..offset + len].to_vec()
            };
            result.insert(name, data);
        }

        Ok(result)
    }

    /// Loads all tensors sequentially (synchronous version).
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// A HashMap mapping all tensor names to their raw data.
    pub fn load_all_tensors_batch_sync(
        &self,
        base_path: impl AsRef<Path>,
    ) -> ReaderResult<HashMap<String, Vec<u8>>> {
        let tensor_names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        self.load_tensors_batch_sync(base_path, &tensor_names)
    }

    async fn validate_single_partition_async(
        &self,
        partition_path: &str,
        entry: &TensorEntry,
    ) -> ReaderResult<()> {
        let required_size = entry
            .offset
            .checked_add(entry.size)
            .ok_or_else(|| ReaderError::ServerlessLlm("offset + size overflow".to_owned()))?;

        // Fast path: check cached size if present
        if let Some(cached_size) = self
            .partition_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(&entry.partition_id).copied())
        {
            if cached_size < required_size {
                return Err(ReaderError::ServerlessLlm(format!(
                    "partition file '{partition_path}' is too small: has {cached_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {} (cached)",
                    entry.offset, entry.size
                )));
            }
            return Ok(());
        }

        let actual_size = stat_partition_size(partition_path).await?;

        if let Ok(mut cache) = self.partition_cache.lock() {
            cache.insert(entry.partition_id, actual_size);
        }

        if actual_size < required_size {
            return Err(ReaderError::ServerlessLlm(format!(
                "partition file '{partition_path}' is too small: has {actual_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {}",
                entry.offset, entry.size
            )));
        }

        Ok(())
    }
    fn validate_single_partition(
        &self,
        partition_path: &str,
        entry: &TensorEntry,
    ) -> ReaderResult<()> {
        let required_size = entry
            .offset
            .checked_add(entry.size)
            .ok_or_else(|| ReaderError::ServerlessLlm("offset + size overflow".to_owned()))?;

        // Check cache first
        {
            let cache = self.partition_cache.lock().unwrap();
            if let Some(&cached_size) = cache.get(&entry.partition_id) {
                if cached_size < required_size {
                    return Err(ReaderError::ServerlessLlm(format!(
                        "partition file '{partition_path}' is too small: has {cached_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {} (cached)",
                        entry.offset, entry.size
                    )));
                }
                return Ok(());
            }
        }

        // Cache miss - perform validation and cache result
        let metadata = std::fs::metadata(partition_path).map_err(|e| {
            // Cache the failure
            self.partition_cache
                .lock()
                .unwrap()
                .insert(entry.partition_id, 0);
            ReaderError::ServerlessLlm(format!(
                "partition file '{partition_path}' not found or inaccessible: {e}"
            ))
        })?;

        let actual_size = metadata.len();

        // Cache the result
        {
            let mut cache = self.partition_cache.lock().unwrap();
            cache.insert(entry.partition_id, actual_size);
        }

        if actual_size < required_size {
            return Err(ReaderError::ServerlessLlm(format!(
                "partition file '{partition_path}' is too small: has {actual_size} bytes, needs at least {required_size} bytes for tensor at offset {} size {}",
                entry.offset, entry.size
            )));
        }

        Ok(())
    }

    /// Returns the partition IDs used by this index.
    #[inline]
    #[must_use]
    pub fn partition_ids(&self) -> Vec<usize> {
        use std::collections::HashSet;
        let mut ids: Vec<_> = self
            .tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        ids.sort_unstable();
        ids
    }
}

#[cfg(target_os = "linux")]
async fn stat_partition_size(path: &str) -> ReaderResult<u64> {
    let stat = statx(path).await.map_err(ReaderError::from)?;
    Ok(stat.stx_size)
}

#[cfg(not(target_os = "linux"))]
async fn stat_partition_size(path: &str) -> ReaderResult<u64> {
    let meta = tokio::fs::metadata(path).await.map_err(ReaderError::from)?;
    Ok(meta.len())
}

impl<'a> IntoIterator for &'a ServerlessLLMIndex {
    type Item = (&'a String, &'a TensorEntry);
    type IntoIter = std::collections::hash_map::Iter<'a, String, TensorEntry>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for ServerlessLLMIndex {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}

impl AsyncReader for ServerlessLLMIndex {
    type Output = Self;

    #[inline]
    async fn load(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let data = backends::load(path.as_ref().to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?)
        .await?;
        parse_index_impl(&data)
    }
}

impl SyncReader for ServerlessLLMIndex {
    type Output = Self;

    #[inline]
    fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self::Output> {
        let data = backends::sync::load(path.as_ref().to_str().ok_or_else(|| {
            ReaderError::InvalidMetadata("path contains invalid UTF-8".to_owned())
        })?)?;
        parse_index_impl(&data)
    }
}

/// View into a memory-mapped tensor with metadata access (lazy loading).
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct TensorMmap {
    /// Memory-mapped tensor data
    mmap: backends::mmap::Mmap,
    /// Tensor metadata
    entry: TensorEntry,
}

#[cfg(target_os = "linux")]
impl TensorMmap {
    /// Creates a new TensorMmap from memory-mapped data.
    #[inline]
    #[must_use]
    pub const fn new(mmap: backends::mmap::Mmap, entry: TensorEntry) -> Self {
        Self { mmap, entry }
    }

    /// Returns the memory-mapped tensor data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.entry.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[i64] {
        &self.entry.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.entry.size
    }
}

/// Owned tensor with data loaded into memory.
///
/// This struct now supports zero-copy access by holding a reference to shared buffers
/// with offset/length metadata instead of copying data for each tensor.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Owned buffer containing tensor data
    data: Vec<u8>,
    /// Offset into the shared buffer where this tensor's data starts
    offset: usize,
    /// Length of this tensor's data in bytes
    len: usize,
    /// Tensor metadata
    entry: TensorEntry,
}

impl Tensor {
    /// Creates a new Tensor from buffer data and metadata.
    #[inline]
    #[must_use]
    pub const fn new(data: Vec<u8>, offset: usize, len: usize, entry: TensorEntry) -> Self {
        Self {
            data,
            offset,
            len,
            entry,
        }
    }

    /// Creates a new Tensor from owned data (for backward compatibility).
    #[inline]
    #[must_use]
    pub const fn from_owned(data: Vec<u8>, entry: TensorEntry) -> Self {
        let len = data.len();
        Self {
            data,
            offset: 0,
            len,
            entry,
        }
    }

    /// Returns the raw tensor data as a slice (zero-copy).
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data[self.offset..self.offset + self.len]
    }

    /// Consumes the view and returns the raw tensor data.
    ///
    /// If this tensor covers the entire buffer, returns the owned Vec<u8>.
    /// Otherwise, returns a copy of the slice.
    #[inline]
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        if self.offset == 0 && self.len == self.data.len() {
            self.data
        } else {
            self.data[self.offset..self.offset + self.len].to_vec()
        }
    }

    /// Returns the tensor's data type.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.entry.dtype
    }

    /// Returns the tensor's shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[i64] {
        &self.entry.shape
    }

    /// Returns the tensor's stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[i64] {
        &self.entry.stride
    }

    /// Returns the tensor's size in bytes.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.entry.size
    }

    /// Returns the partition id containing this tensor.
    #[inline]
    #[must_use]
    pub const fn partition_id(&self) -> usize {
        self.entry.partition_id
    }

    /// Returns the offset into the backing buffer where the tensor begins.
    #[inline]
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the length of the tensor slice within the backing buffer.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns an owned copy of the tensor data slice.
    #[inline]
    #[must_use]
    pub fn to_vec(&self) -> Vec<u8> {
        self.data().to_vec()
    }

    /// Consumes the tensor and returns the owned data slice as a Vec<u8>.
    #[inline]
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        let start = self.offset;
        let end = start + self.len;
        self.data[start..end].to_vec()
    }
}

/// High-level ServerlessLLM reader with owned tensor data.
///
/// This provides a simple API similar to SafeTensors: load once, then access tensors by name.
/// All tensor data is loaded into memory for fast access.
#[derive(Debug, Clone)]
pub struct ServerlessLLM {
    /// All tensors loaded into memory
    tensors: HashMap<String, Tensor>,
}

impl ServerlessLLM {
    /// Loads a ServerlessLLM model from directory asynchronously with eager loading.
    ///
    /// This loads the index file and all tensor data into memory for fast access.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing tensor_index.json and tensor.data_* files
    ///
    /// # Returns
    ///
    /// An `ServerlessLLM` with all tensors loaded and ready for access.
    pub async fn from_directory_async(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = ServerlessLLMIndex::load(&index_path).await?;

        // Load all tensors using async batching with zero-copy
        let tensor_names: Vec<&str> = index.tensors.keys().map(|s| s.as_str()).collect();
        let tensor_data = index
            .load_tensors_batch_zero_copy(&data_path, &tensor_names)
            .await?;

        let tensors = tensor_data
            .into_iter()
            .map(|(name, (buf, offset, len))| {
                let entry = index.get(&name).unwrap().clone();
                (name, Tensor::new(buf, offset, len, entry))
            })
            .collect();

        Ok(Self { tensors })
    }

    /// Loads a ServerlessLLM model from directory with eager loading.
    ///
    /// Loads a ServerlessLLM model from directory synchronously with eager loading.
    ///
    /// This loads the index file and all tensor data into memory for fast access.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing tensor_index.json and tensor.data_* files
    ///
    /// # Returns
    ///
    /// An `ServerlessLLM` with all tensors loaded and ready for access.
    pub fn from_directory(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = ServerlessLLMIndex::load_sync(&index_path)?;

        // Load all tensors
        let tensor_data = index.load_all_tensors_batch_sync(&data_path)?;

        // Convert to Tensor structs with zero-copy support
        let tensors = tensor_data
            .into_iter()
            .map(|(name, data)| {
                let entry = index.get(&name).unwrap().clone();
                (name, Tensor::from_owned(data, entry))
            })
            .collect();

        Ok(Self { tensors })
    }

    /// Returns a reference to the tensor with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the tensor
    ///
    /// # Returns
    ///
    /// Some(&Tensor) if the tensor exists, None otherwise.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Returns tensor names without requiring the `TensorMetadata` trait in scope.
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Returns the number of tensors in the loaded model.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true when no tensors are loaded.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns an immutable view of the loaded tensors.
    #[inline]
    #[must_use]
    pub const fn tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    /// Consumes the model and returns the underlying tensor map.
    #[inline]
    #[must_use]
    pub fn into_tensors(self) -> HashMap<String, Tensor> {
        self.tensors
    }

    /// Returns an iterator over all tensor names and tensors.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }
}

/// Parse `ServerlessLLM` tensor index asynchronously.
#[inline]
pub async fn parse_index(path: impl AsRef<Path>) -> ReaderResult<ServerlessLLMIndex> {
    ServerlessLLMIndex::load(path).await
}

/// Parse `ServerlessLLM` tensor index synchronously (mmap on Linux).
#[inline]
pub fn parse_index_sync(path: impl AsRef<Path>) -> ReaderResult<ServerlessLLMIndex> {
    ServerlessLLMIndex::load_sync(path)
}

/// Load a ServerlessLLM model with eager loading (async).
///
/// This is the high-level API for loading ServerlessLLM models.
/// It loads the index and all tensor data into memory for fast access.
///
/// # Arguments
///
/// * `directory` - Directory containing tensor_index.json and tensor.data_* files
///
/// # Returns
///
/// An `ServerlessLLM` with all tensors loaded and ready for access.
#[inline]
pub async fn load(directory: impl AsRef<Path>) -> ReaderResult<ServerlessLLM> {
    ServerlessLLM::from_directory_async(directory).await
}

/// Load a ServerlessLLM model with eager loading (sync).
///
/// # Arguments
///
/// * `directory` - Directory containing tensor_index.json and tensor.data_* files
///
/// # Returns
///
/// An `ServerlessLLM` with all tensors loaded and ready for access.
#[inline]
pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<ServerlessLLM> {
    ServerlessLLM::from_directory(directory)
}

/// Load a ServerlessLLM model with mmap-based lazy loading (sync).
///
/// This memory-maps all partition files for zero-copy tensor access. Tensor data
/// is loaded lazily on first access.
///
/// # Arguments
///
/// * `directory` - Directory containing tensor_index.json and tensor.data_* files
///
/// # Returns
///
/// A `ServerlessLLMMmap` with all partition files memory-mapped.
///
/// # Errors
///
/// Returns an error if the index file can't be parsed or partition files can't be mapped.
#[inline]
pub fn load_mmap(directory: impl AsRef<Path>) -> ReaderResult<ServerlessLLMMmap> {
    ServerlessLLMMmap::from_directory(directory)
}

pub struct ServerlessLLMMmap {
    /// Index for tensor metadata
    index: ServerlessLLMIndex,
    /// Memory-mapped partition files
    partitions: HashMap<usize, backends::mmap::Mmap>,
}

impl ServerlessLLMMmap {
    /// Loads a ServerlessLLM model from directory synchronously with mmap-based lazy loading.
    ///
    /// This loads the index file and memory-maps all partition files for zero-copy access.
    ///
    /// # Arguments
    ///
    /// * `directory` - Directory containing tensor_index.json and tensor.data_* files
    ///
    /// # Returns
    ///
    /// A `ServerlessLLMMmap` with all partition files memory-mapped.
    pub fn from_directory(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = ServerlessLLMIndex::load_sync(&index_path)?;
        let partition_ids = index.partition_ids();

        // Parallel mmap of all partition files
        let partitions: Result<HashMap<usize, backends::mmap::Mmap>, ReaderError> = partition_ids
            .par_iter()
            .map(|&partition_id| {
                let partition_path = format!("{}_{}", data_path.display(), partition_id);
                let mmap = backends::mmap::map(&partition_path)?;
                Ok((partition_id, mmap))
            })
            .collect();

        Ok(Self {
            index,
            partitions: partitions?,
        })
    }

    /// Returns a lazy view of the tensor with the given name.
    ///
    /// The tensor data is not copied - it's a zero-copy view into the memory-mapped file.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the tensor
    ///
    /// # Returns
    ///
    /// Some(TensorMmap) if the tensor exists, None otherwise.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<TensorMmap> {
        let entry = self.index.get(name)?;
        let mmap = self.partitions.get(&entry.partition_id)?;

        // Create a range view of the mmap for this tensor
        let start = usize::try_from(entry.offset).ok()?;
        let len = usize::try_from(entry.size).ok()?;
        let end = start.checked_add(len)?;

        if end > mmap.len() {
            return None;
        }

        // Create a sub-slice mmap (this is zero-copy)
        let tensor_mmap = backends::mmap::Mmap {
            inner: Arc::clone(&mmap.inner),
            start: mmap.start + start,
            len,
        };

        Some(TensorMmap::new(tensor_mmap, entry.clone()))
    }
}

impl<'a> IntoIterator for &'a ServerlessLLM {
    type Item = (&'a String, &'a Tensor);
    type IntoIter = std::collections::hash_map::Iter<'a, String, Tensor>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for ServerlessLLMMmap {
    #[inline]
    fn len(&self) -> usize {
        self.index.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.index.contains(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.index.tensor_names()
    }
}

impl TensorMetadata for ServerlessLLM {
    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}

/// Core parsing implementation shared by async and sync versions.
pub fn parse_index_impl(data: &[u8]) -> ReaderResult<ServerlessLLMIndex> {
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(data)
        .map_err(|err| ReaderError::ServerlessLlm(format!("JSON parse error: {err}")))?;

    let mut tensors = HashMap::with_capacity(raw.len());

    for (name, value) in raw {
        let arr = value.as_array().ok_or_else(|| {
            ReaderError::ServerlessLlm(format!("tensor entry '{name}' must be an array"))
        })?;

        // Support both 5-field (offset, size, shape, stride, dtype) and
        // 6-field entries where partition_id is present.
        let (offset, size, shape, stride, dtype, partition_id) = match arr.len() {
            5 => {
                let offset = to_u64(arr.first().expect("array has 5 elements"), &name)?;
                let size = to_u64(arr.get(1).expect("array has 5 elements"), &name)?;
                let shape = to_i64_vec(arr.get(2).expect("array has 5 elements"), &name)?;
                let stride = to_i64_vec(arr.get(3).expect("array has 5 elements"), &name)?;
                let dtype = to_string(arr.get(4).expect("array has 5 elements"), &name)?;
                (offset, size, shape, stride, dtype, 0usize)
            }
            6 => {
                let offset = to_u64(arr.first().expect("array has 6 elements"), &name)?;
                let size = to_u64(arr.get(1).expect("array has 6 elements"), &name)?;
                let shape = to_i64_vec(arr.get(2).expect("array has 6 elements"), &name)?;
                let stride = to_i64_vec(arr.get(3).expect("array has 6 elements"), &name)?;
                let dtype = to_string(arr.get(4).expect("array has 6 elements"), &name)?;
                let partition_id = to_usize(arr.get(5).expect("array has 6 elements"), &name)?;
                (offset, size, shape, stride, dtype, partition_id)
            }
            _ => {
                return Err(ReaderError::ServerlessLlm(format!(
                    "tensor entry '{}' must have 5 or 6 elements, got {}",
                    name,
                    arr.len()
                )));
            }
        };

        tensors.insert(
            name,
            TensorEntry {
                offset,
                size,
                shape,
                stride,
                dtype,
                partition_id,
            },
        );
    }

    Ok(ServerlessLLMIndex {
        tensors,
        partition_cache: Mutex::new(HashMap::new()),
    })
}

fn to_u64(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<u64> {
    value.as_u64().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected u64 in tensor '{tensor_name}', got {value:?}"
        ))
    })
}

fn to_usize(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected usize in tensor '{tensor_name}', got {value:?}"
            ))
        })
}

fn to_string(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<String> {
    value
        .as_str()
        .map(std::string::ToString::to_string)
        .ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected string in tensor '{tensor_name}', got {value:?}"
            ))
        })
}

fn to_i64_vec(value: &serde_json::Value, tensor_name: &str) -> ReaderResult<Vec<i64>> {
    let arr = value.as_array().ok_or_else(|| {
        ReaderError::ServerlessLlm(format!(
            "expected array in tensor '{tensor_name}', got {value:?}"
        ))
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(v.as_i64().ok_or_else(|| {
            ReaderError::ServerlessLlm(format!(
                "expected integer in tensor '{tensor_name}', got {v:?}"
            ))
        })?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_index(
        dir: &Path,
        entries: &[(&str, (u64, u64, Vec<i64>, Vec<i64>, &str, usize))],
    ) -> std::path::PathBuf {
        let index_path = dir.join("tensor_index.json");
        let mut root = serde_json::Map::new();

        for (name, (offset, size, shape, stride, dtype, partition_id)) in entries {
            root.insert(
                (*name).to_string(),
                json!([offset, size, shape, stride, *dtype, partition_id]),
            );
        }

        let json_data = serde_json::Value::Object(root).to_string();
        fs::write(&index_path, json_data).expect("write index");
        index_path
    }

    fn write_partition(path: &str, data: &[u8]) {
        let mut file = fs::File::create(path).expect("create partition");
        file.write_all(data).expect("write partition data");
    }

    #[test]
    fn parse_index_impl_supports_partition_defaults_and_ids() {
        let data = br#"{
            "tensor_a": [0, 4, [2,2], [2,1], "f32"],
            "tensor_b": [4, 2, [1,2], [2,1], "i8", 3]
        }"#;

        let index = parse_index_impl(data).expect("parse index");
        let tensor_a = index.get("tensor_a").expect("tensor_a");
        assert_eq!(tensor_a.partition_id, 0);
        assert_eq!(tensor_a.size, 4);
        let tensor_b = index.get("tensor_b").expect("tensor_b");
        assert_eq!(tensor_b.partition_id, 3);
        assert_eq!(tensor_b.shape, vec![1, 2]);
    }

    #[test]
    fn parse_index_impl_rejects_invalid_entry_length() {
        let data = br#"{"bad": [1,2,3,4]}"#;
        let err = parse_index_impl(data).unwrap_err();
        assert!(
            format!("{err}").contains("must have 5 or 6 elements"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn load_tensor_sync_reads_and_caches_partition() {
        let dir = TempDir::new().unwrap();
        let base_path = dir.path().join("tensor.data");
        let part_path = format!("{}_0", base_path.display());

        write_partition(&part_path, b"hello_world");
        let index_path = write_index(
            dir.path(),
            &[("weight", (0, 5, vec![1, 5], vec![5, 1], "u8", 0))],
        );

        let index = parse_index_sync(&index_path).unwrap();
        let first = index
            .load_tensor_sync(&base_path, "weight")
            .expect("first load");
        assert_eq!(first, b"hello");

        // Second load should hit the cached partition metadata path.
        let second = index
            .load_tensor_sync(&base_path, "weight")
            .expect("second load");
        assert_eq!(second, b"hello");
    }

    #[test]
    fn load_tensor_sync_reports_missing_tensor() {
        let dir = TempDir::new().unwrap();
        let base_path = dir.path().join("tensor.data");
        let part_path = format!("{}_0", base_path.display());
        write_partition(&part_path, b"abcde");
        let index_path = write_index(
            dir.path(),
            &[("exists", (0, 5, vec![1, 5], vec![5, 1], "u8", 0))],
        );
        let index = parse_index_sync(&index_path).unwrap();
        let err = index.load_tensor_sync(&base_path, "missing").unwrap_err();
        assert!(format!("{err}").contains("tensor 'missing' not found"));
    }

    #[test]
    fn load_tensor_sync_errors_when_partition_too_small() {
        let dir = TempDir::new().unwrap();
        let base_path = dir.path().join("tensor.data");
        let part_path = format!("{}_0", base_path.display());
        write_partition(&part_path, b"abc"); // shorter than requested size
        let index_path = write_index(
            dir.path(),
            &[("tensor", (0, 5, vec![1, 5], vec![5, 1], "u8", 0))],
        );
        let index = parse_index_sync(&index_path).unwrap();
        let err = index.load_tensor_sync(&base_path, "tensor").unwrap_err();
        assert!(format!("{err}").contains("too small"));
    }

    #[test]
    fn tensor_helpers_expose_data_and_metadata() {
        let entry = TensorEntry {
            offset: 2,
            size: 3,
            shape: vec![3],
            stride: vec![1],
            dtype: "u8".to_owned(),
            partition_id: 0,
        };
        let tensor = Tensor::new(b"012345".to_vec(), 2, 3, entry.clone());
        assert_eq!(tensor.data(), b"234");
        assert_eq!(tensor.to_vec(), b"234");
        assert_eq!(tensor.clone().into_vec(), b"234");
        assert_eq!(tensor.dtype(), "u8");
        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.stride(), &[1]);
        assert_eq!(tensor.size(), 3);
        assert_eq!(tensor.partition_id(), 0);
        assert_eq!(tensor.offset(), 2);
        assert_eq!(tensor.len(), 3);
    }

    #[test]
    fn load_tensors_batch_sync_handles_multiple_partitions() {
        let dir = TempDir::new().unwrap();
        let base_path = dir.path().join("tensor.data");
        write_partition(&format!("{}_0", base_path.display()), b"abcde");
        write_partition(&format!("{}_1", base_path.display()), b"vwxyz");

        let index_path = write_index(
            dir.path(),
            &[
                ("a", (0, 3, vec![3], vec![1], "u8", 0)),
                ("b", (2, 3, vec![3], vec![1], "u8", 1)),
            ],
        );

        let index = parse_index_sync(&index_path).unwrap();
        let tensors = index
            .load_tensors_batch_sync(&base_path, &["a", "b"])
            .unwrap();
        assert_eq!(tensors["a"], b"abc");
        assert_eq!(tensors["b"], b"xyz");

        let mut names = index.tensor_names();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(index.len(), 2);
        assert!(!index.is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn load_mmap_returns_view_within_bounds() {
        let dir = TempDir::new().unwrap();
        let base_path = dir.path().join("tensor.data");
        write_partition(&format!("{}_0", base_path.display()), b"0123456789");
        let _index_path = write_index(dir.path(), &[("slice", (2, 4, vec![4], vec![1], "u8", 0))]);

        let model = load_mmap(dir.path()).expect("load mmap");
        let view = model.tensor("slice").expect("tensor slice");
        assert_eq!(view.data(), b"2345");
        assert_eq!(view.dtype(), "u8");
    }
}
