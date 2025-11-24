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
//! use tensor_store::readers::serverlessllm;
//!
//! // Load entire model with eager loading
//! let model = serverlessllm::load("model_dir/").await?;
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
//! let index = serverlessllm::parse_index("model/tensor_index.json").await?;
//!
//! // Load individual tensors
//! let weight = index.load_tensor("model/tensor.data", "layer.0.weight").await?;
//!
//! // Or load multiple tensors concurrently
//! let tensors = index.load_tensors_batch("model/tensor.data", &["layer.0.weight", "layer.0.bias"]).await?;
//! ```
//!
use crate::backends;
use crate::readers::error::{ReaderError, ReaderResult};
use crate::readers::traits::{AsyncReader, SyncReader, TensorMetadata};
use crate::types::serverlessllm::TensorEntry;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Parsed `ServerlessLLM` index
#[derive(Debug, Default)]
#[non_exhaustive]
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

        let partition_path = format!("{}_{}", base_path.as_ref().display(), entry.partition_id);

        // Validate partition file before loading
        self.validate_single_partition(&partition_path, entry)?;

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

        let partition_path = format!("{}_{}", base_path.as_ref().display(), entry.partition_id);

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

        // Process each partition file with batched I/O
        let mut tensors = HashMap::with_capacity(tensor_names.len());

        for (partition_id, partition_entries) in partition_requests {
            let partition_path = format!("{}_{}", base_path_str, partition_id);

            // Validate all partition files first
            for (_name, entry) in &partition_entries {
                self.validate_single_partition(&partition_path, entry)?;
            }

            // Prepare batch requests for this partition
            let mut batch_requests = Vec::with_capacity(partition_entries.len());
            for (_name, entry) in &partition_entries {
                let len = usize::try_from(entry.size)
                    .map_err(|e| ReaderError::ServerlessLlm(format!("size too large: {e}")))?;
                batch_requests.push((&partition_path, entry.offset, len));
            }

            // Execute batched I/O operation
            let batch_results = backends::load_range_batch(&batch_requests)
                .await
                .map_err(ReaderError::from)?;

            // Map results back to tensor names
            for ((name, _entry), data) in partition_entries.into_iter().zip(batch_results) {
                tensors.insert(name, data);
            }
        }

        Ok(tensors)
    }

    /// Loads all tensors concurrently using io_uring batching.
    ///
    /// This is a convenience method that loads every tensor in the index.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// A HashMap mapping all tensor names to their raw data.
    pub async fn load_all_tensors_batch(
        &self,
        base_path: impl AsRef<Path>,
    ) -> ReaderResult<HashMap<String, Vec<u8>>> {
        let tensor_names: Vec<&str> = self.tensors.keys().map(|s| s.as_str()).collect();
        self.load_tensors_batch(base_path, &tensor_names).await
    }

    /// Loads multiple tensors sequentially (synchronous version).
    ///
    /// Note: This loads tensors one-by-one, not taking advantage of io_uring batching.
    /// For concurrent loading, use the async version.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    /// * `tensor_names` - Names of tensors to load
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor names to their raw data.
    pub fn load_tensors_batch_sync(
        &self,
        base_path: impl AsRef<Path>,
        tensor_names: &[&str],
    ) -> ReaderResult<HashMap<String, Vec<u8>>> {
        let mut tensors = HashMap::with_capacity(tensor_names.len());
        for &name in tensor_names {
            let data = self.load_tensor_sync(base_path.as_ref(), name)?;
            tensors.insert(name.to_string(), data);
        }
        Ok(tensors)
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

    /// Validates that all partition files exist and are readable.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// Ok if all partition files exist, Err with details of missing files.
    #[inline]
    pub fn validate_partitions(&self, base_path: impl AsRef<Path>) -> ReaderResult<()> {
        use std::collections::HashSet;

        // Collect unique partition IDs
        let partition_ids: HashSet<usize> = self
            .tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect();

        let mut missing_partitions = Vec::new();

        for partition_id in partition_ids {
            let partition_path = format!("{}_{}", base_path.as_ref().display(), partition_id);
            if !std::path::Path::new(&partition_path).exists() {
                missing_partitions.push(partition_id);
            }
        }

        if !missing_partitions.is_empty() {
            return Err(ReaderError::ServerlessLlm(format!(
                "missing partition files: {missing_partitions:?}"
            )));
        }

        Ok(())
    }

    /// Validates partition file sizes match expected tensor data.
    ///
    /// # Arguments
    ///
    /// * `base_path` - Base path (without the partition suffix)
    ///
    /// # Returns
    ///
    /// Ok if all partition files have sufficient size, Err with details.
    #[inline]
    pub fn validate_partition_sizes(&self, base_path: impl AsRef<Path>) -> ReaderResult<()> {
        use std::collections::HashMap;

        // Calculate expected size for each partition
        let mut partition_sizes: HashMap<usize, u64> = HashMap::new();
        for entry in self.tensors.values() {
            let max_offset = entry.offset + entry.size;
            partition_sizes
                .entry(entry.partition_id)
                .and_modify(|size| *size = (*size).max(max_offset))
                .or_insert(max_offset);
        }

        let mut errors = Vec::new();

        for (partition_id, expected_size) in partition_sizes {
            let partition_path = format!("{}_{}", base_path.as_ref().display(), partition_id);
            match std::fs::metadata(&partition_path) {
                Ok(metadata) => {
                    let actual_size = metadata.len();
                    if actual_size < expected_size {
                        errors.push(format!(
                            "partition {partition_id}: expected at least {expected_size} bytes, found {actual_size} bytes"
                        ));
                    }
                }
                Err(e) => {
                    errors.push(format!("partition {partition_id}: {e}"));
                }
            }
        }

        if !errors.is_empty() {
            let errors_str = errors.join("; ");
            return Err(ReaderError::ServerlessLlm(format!(
                "partition validation failed: {errors_str}"
            )));
        }

        Ok(())
    }

    /// Returns the total number of unique partition files.
    #[inline]
    #[must_use]
    pub fn partition_count(&self) -> usize {
        use std::collections::HashSet;
        self.tensors
            .values()
            .map(|entry| entry.partition_id)
            .collect::<HashSet<_>>()
            .len()
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
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Tensor {
    /// Raw tensor data
    data: Vec<u8>,
    /// Tensor metadata
    entry: TensorEntry,
}

impl Tensor {
    /// Creates a new Tensor from data and metadata.
    #[inline]
    #[must_use]
    pub const fn new(data: Vec<u8>, entry: TensorEntry) -> Self {
        Self { data, entry }
    }

    /// Returns the raw tensor data.
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consumes the view and returns the raw tensor data.
    #[inline]
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
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

/// High-level ServerlessLLM reader with owned tensor data.
///
/// This provides a simple API similar to SafeTensors: load once, then access tensors by name.
/// All tensor data is loaded into memory for fast access.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ServerlessLLM {
    /// All tensors loaded into memory
    tensors: HashMap<String, Tensor>,
}

impl ServerlessLLM {
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

        // Convert to Tensor structs
        let tensors = tensor_data
            .into_iter()
            .map(|(name, data)| {
                let entry = index.get(&name).unwrap().clone();
                (name, Tensor::new(data, entry))
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
///
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
pub fn load(directory: impl AsRef<Path>) -> ReaderResult<ServerlessLLM> {
    ServerlessLLM::from_directory(directory)
}

/// Load a ServerlessLLM model with mmap-based lazy loading (cross-platform, async).
///
/// This memory-maps all partition files for zero-copy tensor access.
/// Tensor data is loaded lazily on first access.
///
/// # Arguments
///
/// * `directory` - Directory containing tensor_index.json and tensor.data_* files
///
/// # Returns
///
/// A `ServerlessLLMMmap` with all partition files memory-mapped.
///
/// Load a ServerlessLLM model with mmap-based lazy loading (cross-platform, sync).
///
/// This memory-maps all partition files for zero-copy tensor access.
/// Tensor data is loaded lazily on first access.
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

        let mut partitions = HashMap::with_capacity(partition_ids.len());
        for partition_id in partition_ids {
            let partition_path = format!("{}_{}", data_path.display(), partition_id);
            let mmap = backends::mmap::map(&partition_path)?;
            partitions.insert(partition_id, mmap);
        }

        Ok(Self { index, partitions })
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
                let offset = to_u64(arr.get(0).expect("array has 5 elements"), &name)?;
                let size = to_u64(arr.get(1).expect("array has 5 elements"), &name)?;
                let shape = to_i64_vec(arr.get(2).expect("array has 5 elements"), &name)?;
                let stride = to_i64_vec(arr.get(3).expect("array has 5 elements"), &name)?;
                let dtype = to_string(arr.get(4).expect("array has 5 elements"), &name)?;
                (offset, size, shape, stride, dtype, 0usize)
            }
            6 => {
                let offset = to_u64(arr.get(0).expect("array has 6 elements"), &name)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_batch_loading() {
        tokio_uring::start(async {
            let dir = Path::new("test_model_serverlessllm");

            // Test batch loading via index
            let index = ServerlessLLMIndex::load(dir.join("tensor_index.json"))
                .await
                .unwrap();
            let tensor_names = index.tensor_names();
            assert!(!tensor_names.is_empty());

            // Load first few tensors in batch
            let batch_names: Vec<&str> = tensor_names.iter().take(3).cloned().collect();
            let batch_data = index
                .load_tensors_batch(&dir.join("tensor.data"), &batch_names)
                .await
                .unwrap();
            assert_eq!(batch_data.len(), batch_names.len());

            // Load all tensors in batch
            let all_data = index
                .load_all_tensors_batch(&dir.join("tensor.data"))
                .await
                .unwrap();
            assert_eq!(all_data.len(), tensor_names.len());

            // Test ServerlessLLM
            let owned = ServerlessLLM::from_directory(dir).unwrap();
            assert_eq!(owned.len(), tensor_names.len());

            // Test tensor access
            for name in &tensor_names {
                let tensor = owned.tensor(name).unwrap();
                assert!(!tensor.data().is_empty());
            }

            println!("Batch loading test passed! Loaded {} tensors", owned.len());
        });
    }

    #[test]
    fn test_batch_loading_sync() {
        let dir = Path::new("test_model_serverlessllm");

        // Test batch loading via index
        let index = ServerlessLLMIndex::load_sync(dir.join("tensor_index.json")).unwrap();
        let tensor_names = index.tensor_names();
        assert!(!tensor_names.is_empty());

        // Load first few tensors in batch
        let batch_names: Vec<&str> = tensor_names.iter().take(3).cloned().collect();
        let batch_data = index
            .load_tensors_batch_sync(dir.join("tensor.data"), &batch_names)
            .unwrap();
        assert_eq!(batch_data.len(), batch_names.len());

        // Load all tensors in batch
        let all_data = index
            .load_all_tensors_batch_sync(dir.join("tensor.data"))
            .unwrap();
        assert_eq!(all_data.len(), tensor_names.len());

        // Test ServerlessLLM
        let owned = ServerlessLLM::from_directory(dir).unwrap();
        assert_eq!(owned.len(), tensor_names.len());

        // Test tensor access
        for name in &tensor_names {
            let tensor = owned.tensor(name).unwrap();
            assert!(!tensor.data().is_empty());
        }

        println!(
            "Sync batch loading test passed! Loaded {} tensors",
            owned.len()
        );
    }
}
