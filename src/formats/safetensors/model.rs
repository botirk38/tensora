//! `SafeTensors` format model.
//!
//! Public loading is directory-based. Single-file models are supported only when
//! they live in a one-file directory. File-path inputs are rejected.

use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::{Model as ModelTrait, TensorView};
#[cfg(target_os = "linux")]
use crate::storage::availability::{StorageCapabilities, StorageKind};
use crate::storage::buffer::{MmapRegion, OwnedBytes};
#[cfg(target_os = "linux")]
use crate::storage::io_uring::IoUringStorage;
use crate::storage::mmap::MmapStorage;
use crate::storage::sync::SyncStorage;
use crate::storage::tokio::TokioStorage;
use crate::storage::{AsyncReadableStorage, MappableStorage, ReadableStorage};
use rayon::prelude::*;
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors, TensorView as StTensorView};
use safetensors::tensor::{Metadata, TensorInfo};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type TensorShardMap = HashMap<Arc<str>, usize>;
type TensorNameList = Arc<[Arc<str>]>;

/// A parsed safetensors shard with owned storage.
///
/// Stores the parsed [`Metadata`] (header, no lifetime) plus the raw data
/// bytes as an `Arc<[u8]>` beginning at `data_start` within the original
/// buffer. This eliminates the previous `transmute`-based self-referential
/// pattern: tensor data is sliced from `data` on demand using the byte
/// offsets recorded in `Metadata`.
#[derive(Debug, Clone)]
struct OwnedShard {
    metadata: Metadata,
    /// The tensor data region (i.e. the bytes after the safetensors header).
    data: Arc<[u8]>,
}

impl OwnedShard {
    fn from_owned(buffer: OwnedBytes) -> ReaderResult<Self> {
        let bytes: Arc<[u8]> = buffer.into_shared();
        // `read_metadata` returns (header_json_len, metadata); the actual
        // tensor data begins at header_json_len + 8 (the 8-byte length prefix).
        let (header_json_len, metadata) = SafeTensors::read_metadata(&bytes)?;
        const N_LEN: usize = std::mem::size_of::<u64>();
        let data: Arc<[u8]> = Arc::from(&bytes[header_json_len + N_LEN..]);
        Ok(Self { metadata, data })
    }

    fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Self::from_owned(OwnedBytes::from_vec(bytes))
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| ReaderError::TensorNotFound {
                name: name.to_owned(),
            })?;
        tensor_from_info_and_data(info, &self.data, name)
    }
}

/// A parsed safetensors shard backed by a memory-mapped file.
///
/// Stores the parsed [`Metadata`] plus the original [`Mmap`] handle.
/// Tensor data is sliced directly from the mmap on demand — no `transmute`.
///
/// [`MmapRegion`]: crate::storage::buffer::MmapRegion
#[derive(Debug, Clone)]
struct MmapShard {
    metadata: Metadata,
    /// Offset within `mmap` at which tensor data begins.
    data_start: usize,
    mmap: MmapRegion,
}

impl MmapShard {
    fn from_mmap(mmap: MmapRegion) -> ReaderResult<Self> {
        // `read_metadata` returns (header_json_len, metadata); the actual
        // tensor data begins at header_json_len + 8 (the 8-byte length prefix).
        let (header_json_len, metadata) = SafeTensors::read_metadata(mmap.as_slice())?;
        const N_LEN: usize = std::mem::size_of::<u64>();
        let data_start = header_json_len + N_LEN;
        Ok(Self {
            metadata,
            data_start,
            mmap,
        })
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| ReaderError::TensorNotFound {
                name: name.to_owned(),
            })?;
        let data = &self.mmap.as_slice()[self.data_start..];
        tensor_from_info_and_data(info, data, name)
    }
}

// ============================================================================
// Shared helper
// ============================================================================

/// Build a [`Tensor`] from a [`TensorInfo`] and a data slice.
///
/// `data` must be the tensor-data region (i.e. after the safetensors header).
/// `TensorInfo::data_offsets` are relative to the start of that region.
fn tensor_from_info_and_data<'a>(
    info: &TensorInfo,
    data: &'a [u8],
    name: &str,
) -> ReaderResult<Tensor<'a>> {
    let (start, end) = info.data_offsets;
    let slice = data.get(start..end).ok_or_else(|| {
        ReaderError::InvalidMetadata(format!(
            "tensor {name}: data offset {start}..{end} out of bounds (data len {})",
            data.len()
        ))
    })?;
    StTensorView::new(info.dtype, info.shape.clone(), slice)
        .map(Tensor)
        .map_err(ReaderError::from)
}

// ============================================================================
// Owned (eager) model internals
// ============================================================================

#[derive(Debug, Clone)]
struct OwnedSingleModel {
    shard: OwnedShard,
    tensor_names: TensorNameList,
}

impl OwnedSingleModel {
    fn from_shard(shard: OwnedShard) -> Self {
        let mut tensor_names: Vec<Arc<str>> = shard
            .metadata
            .offset_keys()
            .into_iter()
            .map(Arc::from)
            .collect();
        tensor_names.sort_unstable();
        Self {
            shard,
            tensor_names: tensor_names.into(),
        }
    }

    fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Ok(Self::from_shard(OwnedShard::from_bytes(bytes)?))
    }

    fn from_owned(buffer: OwnedBytes) -> ReaderResult<Self> {
        Ok(Self::from_shard(OwnedShard::from_owned(buffer)?))
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        self.shard.tensor(name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }
}

#[derive(Debug, Clone)]
struct OwnedShardedModel {
    shards: Vec<OwnedShard>,
    tensor_shards: TensorShardMap,
    tensor_names: TensorNameList,
}

impl OwnedShardedModel {
    fn from_shards(shards: Vec<OwnedShard>) -> ReaderResult<Self> {
        let (tensor_shards, tensor_names) = Self::build_index(&shards)?;
        Ok(Self {
            shards,
            tensor_shards,
            tensor_names,
        })
    }

    fn build_index(shards: &[OwnedShard]) -> ReaderResult<(TensorShardMap, TensorNameList)> {
        let mut tensor_shards = HashMap::new();
        let mut tensor_names = Vec::new();

        for (shard_idx, shard) in shards.iter().enumerate() {
            for name in shard.metadata.offset_keys() {
                let name: Arc<str> = Arc::from(name.as_str());
                if tensor_shards.insert(name.clone(), shard_idx).is_some() {
                    return Err(ReaderError::InvalidMetadata(format!(
                        "duplicate tensor name across shards: {}",
                        name
                    )));
                }
                tensor_names.push(name);
            }
        }

        tensor_names.sort_unstable();
        Ok((tensor_shards, tensor_names.into()))
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        let shard_idx =
            self.tensor_shards
                .get(name)
                .copied()
                .ok_or_else(|| ReaderError::TensorNotFound {
                    name: name.to_owned(),
                })?;
        self.shards[shard_idx].tensor(name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }

    fn len(&self) -> usize {
        self.tensor_names.len()
    }

    fn is_empty(&self) -> bool {
        self.tensor_names.is_empty()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_shards.contains_key(name)
    }
}

#[derive(Debug, Clone, Copy)]
struct LoadStats {
    shard_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadEngine {
    Sync,
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
}

const MULTI_SHARD_ASYNC_THRESHOLD: u64 = 4 * 1024 * 1024 * 1024;
#[cfg(target_os = "linux")]
const IOURING_SHARD_THRESHOLD: usize = 4;
#[cfg(target_os = "linux")]
const IOURING_BYTE_THRESHOLD: u64 = 8 * 1024 * 1024 * 1024;

impl LoadStats {
    fn choose_engine(&self) -> LoadEngine {
        if self.shard_count <= 1 {
            return LoadEngine::Sync;
        }

        #[cfg(target_os = "linux")]
        {
            self.choose_engine_with_capabilities(StorageCapabilities::cached())
        }
        #[cfg(not(target_os = "linux"))]
        {
            if self.total_bytes >= MULTI_SHARD_ASYNC_THRESHOLD {
                return LoadEngine::TokioAsync;
            }
            LoadEngine::Sync
        }
    }

    #[cfg(target_os = "linux")]
    fn choose_engine_with_capabilities(&self, capabilities: &StorageCapabilities) -> LoadEngine {
        if self.shard_count >= IOURING_SHARD_THRESHOLD
            && self.total_bytes >= IOURING_BYTE_THRESHOLD
            && capabilities.is_available(StorageKind::IoUring)
        {
            return LoadEngine::IoUring;
        }

        if self.total_bytes >= MULTI_SHARD_ASYNC_THRESHOLD
            && capabilities.is_available(StorageKind::Tokio)
        {
            return LoadEngine::TokioAsync;
        }

        LoadEngine::Sync
    }
}

#[derive(Debug, Clone)]
enum ModelStorage {
    Single(OwnedSingleModel),
    Sharded(OwnedShardedModel),
}

/// SafeTensors reader with owned, eager storage.
#[derive(Debug, Clone)]
pub struct Model {
    storage: ModelStorage,
}

impl Model {
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Ok(Self {
            storage: ModelStorage::Single(OwnedSingleModel::from_bytes(bytes)?),
        })
    }

    fn discover_shards(path: impl AsRef<Path>) -> ReaderResult<Vec<PathBuf>> {
        let path = path.as_ref();
        if path.is_file() {
            return Err(ReaderError::InvalidMetadata(format!(
                "SafeTensors loads a directory of .safetensors shards, got file path {}",
                path.display()
            )));
        }
        if !path.is_dir() {
            return Err(ReaderError::InvalidMetadata(format!(
                "SafeTensors loads a directory of .safetensors shards, got file path {}",
                path.display()
            )));
        }

        let mut shard_paths = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if !entry_path.is_file() {
                continue;
            }

            let Some(name) = entry_path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name.ends_with(".safetensors") {
                shard_paths.push(entry_path);
            }
        }

        shard_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if shard_paths.is_empty() {
            return Err(ReaderError::InvalidMetadata(format!(
                "no .safetensors files found in {}",
                path.display()
            )));
        }

        Ok(shard_paths)
    }

    fn directory_stats(path: impl AsRef<Path>) -> ReaderResult<(LoadStats, Vec<PathBuf>)> {
        let shard_paths = Self::discover_shards(path)?;
        let mut total_bytes = 0u64;
        for shard_path in &shard_paths {
            let size = fs::metadata(shard_path)?.len();
            total_bytes = total_bytes.saturating_add(size);
        }
        Ok((
            LoadStats {
                shard_count: shard_paths.len(),
                total_bytes,
            },
            shard_paths,
        ))
    }

    pub async fn load(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let (stats, shard_paths) = Self::directory_stats(&path)?;
        match stats.choose_engine() {
            #[cfg(target_os = "linux")]
            LoadEngine::IoUring => Self::load_io_uring_from_shards(shard_paths),
            LoadEngine::TokioAsync => Self::load_async_from_shards(shard_paths).await,
            LoadEngine::Sync => Self::load_sync_from_shards(shard_paths),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn load_io_uring(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = Self::discover_shards(path)?;
        Self::load_io_uring_from_shards(shard_paths)
    }

    #[cfg(target_os = "linux")]
    fn load_io_uring_from_shards(shard_paths: Vec<PathBuf>) -> ReaderResult<Self> {
        let engine = IoUringStorage::new();
        if shard_paths.len() == 1 {
            let bytes = engine
                .read_file(&shard_paths[0])
                .map_err(ReaderError::from)?;
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(bytes)?),
            });
        }

        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_par_iter()
            .map(|path| {
                let bytes = engine.read_file(&path).map_err(ReaderError::from)?;
                OwnedShard::from_owned(bytes)
            })
            .collect();
        Ok(Self {
            storage: ModelStorage::Sharded(OwnedShardedModel::from_shards(shards?)?),
        })
    }

    pub async fn load_async(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = Self::discover_shards(path)?;
        Self::load_async_from_shards(shard_paths).await
    }

    async fn load_async_from_shards(shard_paths: Vec<PathBuf>) -> ReaderResult<Self> {
        let engine = TokioStorage::new();
        if shard_paths.len() == 1 {
            let bytes = engine
                .read_file(&shard_paths[0])
                .await
                .map_err(ReaderError::from)?;
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(bytes)?),
            });
        }

        let mut total_bytes = 0u64;
        let mut max_shard_bytes = 0u64;
        for shard_path in &shard_paths {
            if let Ok(size) = fs::metadata(shard_path).map(|m| m.len()) {
                total_bytes = total_bytes.saturating_add(size);
                max_shard_bytes = max_shard_bytes.max(size);
            }
        }

        // Compute a chunk size that balances concurrency against memory pressure.
        // More shards and larger average sizes warrant higher concurrency; skew
        // (one very large shard) reduces it to avoid head-of-line stalls.
        const PARALLELISM_TARGET_BYTES: f64 = 256.0 * 1024.0 * 1024.0;
        let limit = {
            let n = shard_paths.len();
            let total = total_bytes.max(1) as f64;
            let max_item = max_shard_bytes.max(1) as f64;
            let avg_item = total / n as f64;
            let count_factor = (n as f64).ln_1p();
            let size_factor = (PARALLELISM_TARGET_BYTES / avg_item).sqrt().clamp(0.5, 4.0);
            let skew_factor = (max_item / avg_item).sqrt().clamp(1.0, 4.0);
            let raw = (1.0 + count_factor * size_factor / skew_factor).clamp(1.0, n as f64);
            let cpu = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
                .max(2);
            (raw as usize).min(cpu).max(n.clamp(2, 4).min(cpu))
        };
        let mut shard_bytes: Vec<OwnedBytes> = Vec::new();
        for chunk in shard_paths.chunks(limit) {
            for path in chunk {
                let bytes = engine.read_file(path).await.map_err(ReaderError::from)?;
                shard_bytes.push(bytes);
            }
        }

        let shards: ReaderResult<Vec<_>> = shard_bytes
            .into_par_iter()
            .map(OwnedShard::from_owned)
            .collect();
        Ok(Self {
            storage: ModelStorage::Sharded(OwnedShardedModel::from_shards(shards?)?),
        })
    }

    pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = Self::discover_shards(path)?;
        Self::load_sync_from_shards(shard_paths)
    }

    fn load_sync_from_shards(shard_paths: Vec<PathBuf>) -> ReaderResult<Self> {
        let engine = SyncStorage::new();
        if shard_paths.len() == 1 {
            let bytes = engine
                .read_file(&shard_paths[0])
                .map_err(ReaderError::from)?;
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(bytes)?),
            });
        }

        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_par_iter()
            .map(|path| {
                let bytes = engine.read_file(&path).map_err(ReaderError::from)?;
                OwnedShard::from_owned(bytes)
            })
            .collect();
        Ok(Self {
            storage: ModelStorage::Sharded(OwnedShardedModel::from_shards(shards?)?),
        })
    }

    #[inline]
    pub fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        match &self.storage {
            ModelStorage::Single(model) => model.tensor(name),
            ModelStorage::Sharded(model) => model.tensor(name),
        }
    }

    #[inline]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        match &self.storage {
            ModelStorage::Single(model) => model.tensor_names(),
            ModelStorage::Sharded(model) => model.tensor_names(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.storage {
            ModelStorage::Single(model) => model.tensor_names().len(),
            ModelStorage::Sharded(model) => model.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.storage {
            ModelStorage::Single(model) => model.tensor_names().is_empty(),
            ModelStorage::Sharded(model) => model.is_empty(),
        }
    }

    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        match &self.storage {
            ModelStorage::Single(model) => model.tensor(name).is_ok(),
            ModelStorage::Sharded(model) => model.contains(name),
        }
    }
}

impl ModelTrait for Model {
    type Tensor<'a>
        = Tensor<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        Model::len(self)
    }

    fn contains(&self, name: &str) -> bool {
        Model::contains(self, name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        Model::tensor_names(self)
    }

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        Model::tensor(self, name).ok()
    }
}

#[derive(Debug, Clone)]
struct MmapSingleModel {
    shard: MmapShard,
    tensor_names: TensorNameList,
}

impl MmapSingleModel {
    fn from_shard(shard: MmapShard) -> Self {
        let mut tensor_names: Vec<Arc<str>> = shard
            .metadata
            .offset_keys()
            .into_iter()
            .map(Arc::from)
            .collect();
        tensor_names.sort_unstable();
        Self {
            shard,
            tensor_names: tensor_names.into(),
        }
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        self.shard.tensor(name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }
}

#[derive(Debug, Clone)]
struct MmapShardedModel {
    shards: Vec<MmapShard>,
    tensor_shards: TensorShardMap,
    tensor_names: TensorNameList,
}

impl MmapShardedModel {
    fn from_shards(shards: Vec<MmapShard>) -> ReaderResult<Self> {
        let mut tensor_shards = HashMap::new();
        let mut tensor_names = Vec::new();

        for (shard_idx, shard) in shards.iter().enumerate() {
            for name in shard.metadata.offset_keys() {
                let name: Arc<str> = Arc::from(name.as_str());
                if tensor_shards.insert(name.clone(), shard_idx).is_some() {
                    return Err(ReaderError::InvalidMetadata(format!(
                        "duplicate tensor name across shards: {}",
                        name
                    )));
                }
                tensor_names.push(name);
            }
        }

        tensor_names.sort_unstable();
        Ok(Self {
            shards,
            tensor_shards,
            tensor_names: tensor_names.into(),
        })
    }

    fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        let shard_idx =
            self.tensor_shards
                .get(name)
                .copied()
                .ok_or_else(|| ReaderError::TensorNotFound {
                    name: name.to_owned(),
                })?;
        self.shards[shard_idx].tensor(name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }

    fn len(&self) -> usize {
        self.tensor_names.len()
    }

    fn is_empty(&self) -> bool {
        self.tensor_names.is_empty()
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_shards.contains_key(name)
    }
}

#[derive(Debug, Clone)]
enum MmapModelStorage {
    Single(MmapSingleModel),
    Sharded(MmapShardedModel),
}

/// SafeTensors reader with mmap-backed, lazy storage.
#[derive(Debug, Clone)]
pub struct MmapModel {
    storage: MmapModelStorage,
}

impl MmapModel {
    pub fn open(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = Model::discover_shards(path)?;
        let mapper = MmapStorage::new();
        if shard_paths.len() == 1 {
            let mmap = mapper
                .map_file(&shard_paths[0])
                .map_err(ReaderError::from)?;
            return Ok(Self {
                storage: MmapModelStorage::Single(MmapSingleModel::from_shard(
                    MmapShard::from_mmap(mmap)?,
                )),
            });
        }

        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_par_iter()
            .map(|shard_path| {
                let mmap = mapper.map_file(&shard_path).map_err(ReaderError::from)?;
                MmapShard::from_mmap(mmap)
            })
            .collect();
        Ok(Self {
            storage: MmapModelStorage::Sharded(MmapShardedModel::from_shards(shards?)?),
        })
    }

    #[inline]
    pub fn tensor<'a>(&'a self, name: &str) -> ReaderResult<Tensor<'a>> {
        match &self.storage {
            MmapModelStorage::Single(model) => model.tensor(name),
            MmapModelStorage::Sharded(model) => model.tensor(name),
        }
    }

    #[inline]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        match &self.storage {
            MmapModelStorage::Single(model) => model.tensor_names(),
            MmapModelStorage::Sharded(model) => model.tensor_names(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.storage {
            MmapModelStorage::Single(model) => model.tensor_names().len(),
            MmapModelStorage::Sharded(model) => model.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.storage {
            MmapModelStorage::Single(model) => model.tensor_names().is_empty(),
            MmapModelStorage::Sharded(model) => model.is_empty(),
        }
    }

    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        match &self.storage {
            MmapModelStorage::Single(model) => model.tensor(name).is_ok(),
            MmapModelStorage::Sharded(model) => model.contains(name),
        }
    }
}

impl ModelTrait for MmapModel {
    type Tensor<'a>
        = Tensor<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        MmapModel::len(self)
    }

    fn contains(&self, name: &str) -> bool {
        MmapModel::contains(self, name)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        MmapModel::tensor_names(self)
    }

    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        MmapModel::tensor(self, name).ok()
    }
}

/// Newtype wrapper around safetensors tensor view to implement `TensorView`.
pub struct Tensor<'a>(pub(super) safetensors::tensor::TensorView<'a>);

impl<'a> Tensor<'a> {
    #[inline]
    #[must_use]
    fn dtype_str(dtype: &Dtype) -> &'static str {
        match dtype {
            Dtype::BOOL => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::I32 => "I32",
            Dtype::I64 => "I64",
            Dtype::U32 => "U32",
            Dtype::U64 => "U64",
            Dtype::BF16 => "BF16",
            _ => "UNKNOWN",
        }
    }
}

impl<'a> TensorView for Tensor<'a> {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn dtype(&self) -> &str {
        Self::dtype_str(&self.0.dtype())
    }

    fn data(&self) -> &[u8] {
        self.0.data()
    }
}

impl TryFrom<Vec<u8>> for Model {
    type Error = ReaderError;

    fn try_from(bytes: Vec<u8>) -> Result<Self, Self::Error> {
        Self::from_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::traits::TensorView;
    #[cfg(target_os = "linux")]
    use crate::storage::availability::{StorageAvailability, UnavailableReason};
    use safetensors::serialize;
    use safetensors::tensor::TensorView as StTensorView;
    use tempfile::TempDir;

    fn sample_bytes() -> Vec<u8> {
        let data = vec![1u8, 2, 3, 4];
        let view = StTensorView::new(Dtype::U8, vec![4], &data).expect("create tensor view");
        serialize([("tensor", view)], None).expect("serialize")
    }

    fn write_shard(path: &Path, tensors: Vec<(&str, StTensorView<'_>)>) {
        let bytes = serialize(tensors, None).expect("serialize shard");
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn owned_from_bytes_parses() {
        let owned = Model::from_bytes(sample_bytes()).expect("parse");
        assert_eq!(owned.len(), 1);
        assert_eq!(owned.tensor_names()[0].as_ref(), "tensor");
        assert_eq!(owned.tensor("tensor").unwrap().shape(), &[4]);
    }

    #[test]
    fn load_sync_rejects_file_paths() {
        let err = Model::load_sync("/tmp/not-a-directory.safetensors").unwrap_err();
        assert!(matches!(err, ReaderError::InvalidMetadata(_)));
    }

    #[test]
    fn load_sync_supports_one_file_directory() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let model = Model::load_sync(dir.path()).expect("load sync");
        assert_eq!(model.len(), 1);
        assert!(model.contains("tensor"));
        assert_eq!(model.tensor("tensor").unwrap().dtype(), "U8");
    }

    #[test]
    fn load_sync_supports_multi_shard_directory() {
        let dir = TempDir::new().unwrap();
        let shard1 = dir.path().join("model-00001-of-00002.safetensors");
        let shard2 = dir.path().join("model-00002-of-00002.safetensors");

        let view1 = StTensorView::new(Dtype::U8, vec![2], &[1u8, 2]).unwrap();
        let view2 = StTensorView::new(Dtype::U8, vec![3], &[3u8, 4, 5]).unwrap();
        write_shard(&shard1, vec![("a", view1)]);
        write_shard(&shard2, vec![("b", view2)]);

        let model = Model::load_sync(dir.path()).expect("load shards");
        assert_eq!(model.len(), 2);
        assert!(model.contains("a"));
        assert!(model.contains("b"));
        assert_eq!(model.tensor("a").unwrap().shape(), &[2]);
        assert_eq!(model.tensor("b").unwrap().shape(), &[3]);
    }

    #[test]
    fn mmap_loader_supports_directory() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let mmap = MmapModel::open(dir.path()).expect("open mmap");
        assert_eq!(mmap.len(), 1);
        assert!(mmap.contains("tensor"));
        assert_eq!(mmap.tensor("tensor").unwrap().dtype(), "U8");
    }

    #[test]
    fn choose_sync_for_single_large_shard() {
        let stats = LoadStats {
            shard_count: 1,
            total_bytes: 2 * 1024 * 1024 * 1024,
        };
        assert_eq!(stats.choose_engine(), LoadEngine::Sync);
    }

    #[test]
    fn choose_sync_for_many_small_shards() {
        let stats = LoadStats {
            shard_count: 16,
            total_bytes: 256 * 1024 * 1024,
        };

        match stats.choose_engine() {
            LoadEngine::Sync => {}
            LoadEngine::TokioAsync => {}
            #[cfg(target_os = "linux")]
            LoadEngine::IoUring => panic!("expected Sync, got IoUring"),
        }
    }

    #[test]
    fn selector_boundary_single_shard_large() {
        let stats = LoadStats {
            shard_count: 1,
            total_bytes: 8 * 1024 * 1024 * 1024,
        };
        assert_eq!(stats.choose_engine(), LoadEngine::Sync);
    }

    #[test]
    fn selector_boundary_two_shard_small() {
        let stats = LoadStats {
            shard_count: 2,
            total_bytes: 1024 * 1024 * 1024,
        };
        match stats.choose_engine() {
            LoadEngine::Sync => {}
            LoadEngine::TokioAsync => {}
            #[cfg(target_os = "linux")]
            LoadEngine::IoUring => panic!("expected Sync for small 2-shard"),
        }
    }

    #[test]
    fn selector_boundary_four_shard_large() {
        let stats = LoadStats {
            shard_count: 4,
            total_bytes: 8 * 1024 * 1024 * 1024,
        };
        match stats.choose_engine() {
            LoadEngine::Sync => {}
            LoadEngine::TokioAsync => {}
            #[cfg(target_os = "linux")]
            LoadEngine::IoUring => {}
        }
    }

    #[cfg(target_os = "linux")]
    fn capabilities_with_io_uring(io_uring: StorageAvailability) -> StorageCapabilities {
        StorageCapabilities {
            sync: StorageAvailability::Available,
            tokio: StorageAvailability::Available,
            mmap: StorageAvailability::Available,
            io_uring,
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selector_uses_async_for_large_safetensors_when_io_uring_unavailable() {
        let stats = LoadStats {
            shard_count: 5,
            total_bytes: 16 * 1024 * 1024 * 1024,
        };
        let capabilities = capabilities_with_io_uring(StorageAvailability::unavailable(
            UnavailableReason::PermissionDenied,
            "io_uring_setup returned EPERM",
        ));

        assert_eq!(
            stats.choose_engine_with_capabilities(&capabilities),
            LoadEngine::TokioAsync
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selector_keeps_sync_for_small_safetensors_when_io_uring_unavailable() {
        let stats = LoadStats {
            shard_count: 2,
            total_bytes: 1024 * 1024 * 1024,
        };
        let capabilities = capabilities_with_io_uring(StorageAvailability::unavailable(
            UnavailableReason::PermissionDenied,
            "io_uring_setup returned EPERM",
        ));

        assert_eq!(
            stats.choose_engine_with_capabilities(&capabilities),
            LoadEngine::Sync
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn selector_boundary_four_shard_small() {
        let stats = LoadStats {
            shard_count: 4,
            total_bytes: 2 * 1024 * 1024 * 1024,
        };
        match stats.choose_engine() {
            LoadEngine::Sync => {}
            LoadEngine::TokioAsync => {}
            LoadEngine::IoUring => panic!("expected Sync below threshold"),
        }
    }

    #[test]
    fn load_async_one_shard() {
        use crate::test_utils::run_async;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let model = run_async(Model::load(dir.path())).expect("load async");
        assert_eq!(model.len(), 1);
        assert!(model.contains("tensor"));
    }

    #[test]
    fn load_async_multi_shard() {
        use crate::test_utils::run_async;
        let dir = TempDir::new().unwrap();
        let s1 = dir.path().join("model-00001-of-00002.safetensors");
        let s2 = dir.path().join("model-00002-of-00002.safetensors");
        let v1 = StTensorView::new(Dtype::U8, vec![2], &[1u8, 2]).unwrap();
        let v2 = StTensorView::new(Dtype::U8, vec![3], &[3u8, 4, 5]).unwrap();
        write_shard(&s1, vec![("a", v1)]);
        write_shard(&s2, vec![("b", v2)]);

        let model = run_async(Model::load(dir.path())).expect("load async multi");
        assert_eq!(model.len(), 2);
        assert!(model.contains("a"));
        assert!(model.contains("b"));
    }

    #[test]
    fn load_sync_rejects_empty_directory() {
        let dir = TempDir::new().unwrap();
        let err = Model::load_sync(dir.path()).unwrap_err();
        assert!(matches!(err, ReaderError::InvalidMetadata(_)));
    }

    #[test]
    fn from_bytes_corrupted_data() {
        let result = Model::from_bytes(vec![0xFF; 64]);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_returns_none_for_missing() {
        let model = Model::from_bytes(sample_bytes()).unwrap();
        assert!(model.tensor("nonexistent").is_err());
    }

    #[test]
    fn mmap_model_multi_shard() {
        let dir = TempDir::new().unwrap();
        let s1 = dir.path().join("model-00001-of-00002.safetensors");
        let s2 = dir.path().join("model-00002-of-00002.safetensors");
        let v1 = StTensorView::new(Dtype::U8, vec![2], &[1u8, 2]).unwrap();
        let v2 = StTensorView::new(Dtype::U8, vec![3], &[3u8, 4, 5]).unwrap();
        write_shard(&s1, vec![("a", v1)]);
        write_shard(&s2, vec![("b", v2)]);

        let mmap = MmapModel::open(dir.path()).unwrap();
        assert_eq!(mmap.len(), 2);
        assert!(mmap.contains("a"));
        assert!(mmap.contains("b"));
    }

    #[test]
    fn mmap_model_tensor_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        std::fs::write(&path, sample_bytes()).unwrap();

        let mmap = MmapModel::open(dir.path()).unwrap();
        assert!(mmap.tensor("nonexistent").is_err());
    }

    #[test]
    fn model_tensor_data_matches() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("model.safetensors");
        let data = vec![10u8, 20, 30, 40];
        let view = StTensorView::new(Dtype::U8, vec![4], &data).unwrap();
        let bytes = serialize([("w", view)], None).unwrap();
        std::fs::write(&path, bytes).unwrap();

        let model = Model::load_sync(dir.path()).unwrap();
        let t = model.tensor("w").unwrap();
        assert_eq!(t.data(), &[10, 20, 30, 40]);
    }
}
