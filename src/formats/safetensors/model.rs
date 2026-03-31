//! `SafeTensors` format model.
//!
//! Public loading is directory-based. Single-file models are supported only when
//! they live in a one-file directory. File-path inputs are rejected.

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::{Model as ModelTrait, TensorView};
use futures::future::try_join_all;
use rayon::prelude::*;
pub use safetensors::SafeTensorError;
pub use safetensors::tensor::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type TensorShardMap = HashMap<Arc<str>, usize>;
type TensorNameList = Arc<[Arc<str>]>;

#[inline]
#[must_use]
fn dtype_to_str(dtype: &Dtype) -> &'static str {
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

#[inline]
fn invalid_metadata(msg: impl Into<String>) -> ReaderError {
    ReaderError::InvalidMetadata(msg.into())
}

fn dir_only_error(path: &Path) -> ReaderError {
    invalid_metadata(format!(
        "SafeTensors loads a directory of .safetensors shards, got file path {}",
        path.display()
    ))
}

fn discover_shard_paths(dir: &Path) -> ReaderResult<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(dir_only_error(dir));
    }

    let mut shard_paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".safetensors") {
            shard_paths.push(path);
        }
    }

    shard_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if shard_paths.is_empty() {
        return Err(invalid_metadata(format!(
            "no .safetensors files found in {}",
            dir.display()
        )));
    }

    Ok(shard_paths)
}

fn normalize_directory(path: impl AsRef<Path>) -> ReaderResult<Vec<PathBuf>> {
    let path = path.as_ref();
    if path.is_file() {
        return Err(dir_only_error(path));
    }

    discover_shard_paths(path)
}

#[derive(Debug, Clone, Copy)]
struct LoadStats {
    shard_count: usize,
    total_bytes: u64,
    max_shard_bytes: u64,
}

const SYNC_BASE_COST_NS: f64 = 180_000.0;
const ASYNC_BASE_COST_NS: f64 = 2_000_000.0;
const SYNC_PER_SHARD_COST_NS: f64 = 35_000.0;
const ASYNC_PER_SHARD_COST_NS: f64 = 150_000_000.0;
const THROUGHPUT_BPS: f64 = 7.5 * 1024.0 * 1024.0 * 1024.0;
const PARALLELISM_TARGET_BYTES: f64 = 128.0 * 1024.0 * 1024.0;

#[cfg(target_os = "linux")]
const IO_URING_BASE_COST_NS: f64 = 500_000.0;
#[cfg(target_os = "linux")]
const IO_URING_THROUGHPUT_BPS: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;
#[cfg(target_os = "linux")]
const IO_URING_PER_SHARD_COST_NS: f64 = 50_000.0;

fn bytes_to_ns(bytes: u64, throughput_bps: f64) -> f64 {
    (bytes as f64 / throughput_bps) * 1_000_000_000.0
}

fn effective_async_parallelism(stats: &LoadStats) -> f64 {
    if stats.shard_count <= 1 {
        return 1.0;
    }

    let avg_shard_bytes = stats.total_bytes.max(1) as f64 / stats.shard_count as f64;
    let max_shard_bytes = stats.max_shard_bytes.max(1) as f64;
    let count_factor = (stats.shard_count as f64).ln_1p();
    let size_factor = (PARALLELISM_TARGET_BYTES / avg_shard_bytes)
        .sqrt()
        .clamp(0.5, 4.0);
    let skew_factor = (max_shard_bytes / avg_shard_bytes).sqrt().clamp(1.0, 4.0);

    (1.0 + count_factor * size_factor / skew_factor).clamp(1.0, stats.shard_count as f64)
}

fn estimate_sync_cost(stats: &LoadStats) -> f64 {
    SYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS)
        + SYNC_PER_SHARD_COST_NS * stats.shard_count as f64
}

fn estimate_async_cost(stats: &LoadStats) -> f64 {
    let parallelism = effective_async_parallelism(stats);

    ASYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS) / parallelism
        + ASYNC_PER_SHARD_COST_NS * stats.shard_count as f64
}

#[cfg(target_os = "linux")]
fn estimate_io_uring_cost(stats: &LoadStats) -> f64 {
    IO_URING_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, IO_URING_THROUGHPUT_BPS)
        + IO_URING_PER_SHARD_COST_NS * stats.shard_count as f64
}

enum LoadBackend {
    Sync,
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
}

fn choose_load_backend(stats: &LoadStats) -> LoadBackend {
    #[cfg(target_os = "linux")]
    {
        if stats.shard_count <= 1 {
            let sync_cost = estimate_sync_cost(stats);
            let io_uring_cost = estimate_io_uring_cost(stats);
            if io_uring_cost < sync_cost {
                return LoadBackend::IoUring;
            }
            return LoadBackend::Sync;
        }
        let sync_cost = estimate_sync_cost(stats);
        let async_cost = estimate_async_cost(stats);
        let io_uring_cost = estimate_io_uring_cost(stats);
        if io_uring_cost < sync_cost && io_uring_cost < async_cost {
            return LoadBackend::IoUring;
        }
        if async_cost < sync_cost {
            return LoadBackend::TokioAsync;
        }
        LoadBackend::Sync
    }
    #[cfg(not(target_os = "linux"))]
    {
        if stats.shard_count <= 1 {
            return LoadBackend::Sync;
        }
        if estimate_async_cost(stats) < estimate_sync_cost(stats) {
            LoadBackend::TokioAsync
        } else {
            LoadBackend::Sync
        }
    }
}

#[derive(Debug)]
struct OwnedShard {
    tensors: SafeTensors<'static>,
    buffer: backends::byte::OwnedBytes,
}

impl OwnedShard {
    fn from_owned(buffer: backends::byte::OwnedBytes) -> ReaderResult<Self> {
        let slice: &[u8] = &buffer;
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;
        Ok(Self { tensors, buffer })
    }

    fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Self::from_owned(backends::byte::OwnedBytes::Shared(bytes.into()))
    }
}

#[derive(Debug)]
struct MmapShard {
    tensors: SafeTensors<'static>,
    _mmap: backends::mmap::Mmap,
}

impl MmapShard {
    fn from_mmap(mmap: backends::mmap::Mmap) -> ReaderResult<Self> {
        let slice = mmap.as_slice();
        let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
        let tensors = SafeTensors::deserialize(static_slice)?;
        Ok(Self {
            tensors,
            _mmap: mmap,
        })
    }
}

impl Clone for MmapShard {
    fn clone(&self) -> Self {
        Self::from_mmap(self._mmap.clone()).expect("valid safetensors mmap data")
    }
}

fn build_tensor_index<'a>(
    shards: impl Iterator<Item = &'a SafeTensors<'static>>,
) -> ReaderResult<(TensorShardMap, TensorNameList)> {
    let mut tensor_shards = HashMap::new();
    let mut tensor_names = Vec::new();

    for (shard_idx, tensors) in shards.enumerate() {
        for name in tensors.names() {
            let name: Arc<str> = name.into();
            if tensor_shards.insert(name.clone(), shard_idx).is_some() {
                return Err(invalid_metadata(format!(
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

async fn load_bytes_async(path: &Path) -> ReaderResult<backends::byte::OwnedBytes> {
    let mut reader = backends::AsyncReader::new();
    reader.load(path).await.map_err(Into::into)
}

fn load_bytes_sync(path: &Path) -> ReaderResult<backends::byte::OwnedBytes> {
    let mut reader = backends::SyncReader::new();
    reader.load(path).map_err(Into::into)
}

#[cfg(target_os = "linux")]
fn load_bytes_io_uring(reader: &mut backends::io_uring::Reader, path: &Path) -> ReaderResult<backends::byte::OwnedBytes> {
    reader.load(path).map_err(Into::into)
}

fn load_mmap(path: &Path) -> ReaderResult<MmapShard> {
    let path_str = path
        .to_str()
        .ok_or_else(|| invalid_metadata("path contains invalid UTF-8".to_owned()))?;
    let mmap = backends::mmap::map(path_str)?;
    MmapShard::from_mmap(mmap)
}

#[derive(Debug)]
struct OwnedSingleModel {
    shard: OwnedShard,
    tensor_names: TensorNameList,
}

impl OwnedSingleModel {
    fn from_shard(shard: OwnedShard) -> Self {
        let mut tensor_names: Vec<Arc<str>> =
            shard.tensors.names().into_iter().map(Into::into).collect();
        tensor_names.sort_unstable();
        Self {
            shard,
            tensor_names: tensor_names.into(),
        }
    }

    fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Ok(Self::from_shard(OwnedShard::from_bytes(bytes)?))
    }

    fn from_owned(buffer: backends::byte::OwnedBytes) -> ReaderResult<Self> {
        Ok(Self::from_shard(OwnedShard::from_owned(buffer)?))
    }

    fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        self.shard
            .tensors
            .tensor(name)
            .map(Tensor)
            .map_err(ReaderError::from)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }
}

#[derive(Debug)]
struct OwnedShardedModel {
    shards: Vec<OwnedShard>,
    tensor_shards: TensorShardMap,
    tensor_names: TensorNameList,
}

impl OwnedShardedModel {
    fn from_shards(shards: Vec<OwnedShard>) -> ReaderResult<Self> {
        let (tensor_shards, tensor_names) = build_tensor_index(shards.iter().map(|s| &s.tensors))?;
        Ok(Self {
            shards,
            tensor_shards,
            tensor_names,
        })
    }

    fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        let shard_idx =
            self.tensor_shards
                .get(name)
                .copied()
                .ok_or_else(|| ReaderError::TensorNotFound {
                    name: name.to_owned(),
                })?;
        self.shards[shard_idx]
            .tensors
            .tensor(name)
            .map(Tensor)
            .map_err(ReaderError::from)
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

#[derive(Debug)]
enum ModelStorage {
    Single(OwnedSingleModel),
    Sharded(OwnedShardedModel),
}

/// SafeTensors reader with owned, eager storage.
#[derive(Debug)]
pub struct Model {
    storage: ModelStorage,
}

impl Model {
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> ReaderResult<Self> {
        Ok(Self {
            storage: ModelStorage::Single(OwnedSingleModel::from_bytes(bytes)?),
        })
    }

    fn directory_stats(path: impl AsRef<Path>) -> ReaderResult<LoadStats> {
        let shard_paths = normalize_directory(path)?;
        let mut total_bytes = 0u64;
        let mut max_shard_bytes = 0u64;
        for shard_path in &shard_paths {
            let size = fs::metadata(shard_path)?.len();
            total_bytes = total_bytes.saturating_add(size);
            max_shard_bytes = max_shard_bytes.max(size);
        }
        Ok(LoadStats {
            shard_count: shard_paths.len(),
            total_bytes,
            max_shard_bytes,
        })
    }

    pub async fn load(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let stats = Self::directory_stats(&path)?;
        match choose_load_backend(&stats) {
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => Self::load_io_uring(path),
            LoadBackend::TokioAsync => Self::load_async(path).await,
            LoadBackend::Sync => Self::load_sync(path),
        }
    }

    #[cfg(target_os = "linux")]
    pub fn load_io_uring(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = normalize_directory(path)?;
        if shard_paths.len() == 1 {
            let mut reader = backends::io_uring::Reader::new()?;
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(
                    load_bytes_io_uring(&mut reader, &shard_paths[0])?,
                )?),
            });
        }

        let mut reader = backends::io_uring::Reader::new()?;
        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_iter()
            .map(|shard_path| OwnedShard::from_owned(load_bytes_io_uring(&mut reader, &shard_path)?))
            .collect();
        Ok(Self {
            storage: ModelStorage::Sharded(OwnedShardedModel::from_shards(shards?)?),
        })
    }

    pub async fn load_async(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = normalize_directory(path)?;
        if shard_paths.len() == 1 {
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(
                    load_bytes_async(&shard_paths[0]).await?,
                )?),
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

        let limit = backends::bounded_async_concurrency(
            shard_paths.len(),
            total_bytes,
            max_shard_bytes,
        );
        let mut shard_bytes: Vec<backends::byte::OwnedBytes> = Vec::new();

        for chunk in shard_paths.chunks(limit) {
            let chunk_results = try_join_all(
                chunk
                    .iter()
                    .map(|sp| async move { load_bytes_async(sp).await }),
            )
            .await?;
            shard_bytes.extend(chunk_results);
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
        let shard_paths = normalize_directory(path)?;
        if shard_paths.len() == 1 {
            return Ok(Self {
                storage: ModelStorage::Single(OwnedSingleModel::from_owned(load_bytes_sync(
                    &shard_paths[0],
                )?)?),
            });
        }

        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_par_iter()
            .map(|shard_path| OwnedShard::from_owned(load_bytes_sync(&shard_path)?))
            .collect();
        Ok(Self {
            storage: ModelStorage::Sharded(OwnedShardedModel::from_shards(shards?)?),
        })
    }

    #[inline]
    pub fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
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

impl Clone for Model {
    fn clone(&self) -> Self {
        match &self.storage {
            ModelStorage::Single(model) => Self {
                storage: ModelStorage::Single(OwnedSingleModel {
                    shard: OwnedShard::from_bytes(model.shard.buffer.to_vec())
                        .expect("valid safetensors data"),
                    tensor_names: model.tensor_names.clone(),
                }),
            },
            ModelStorage::Sharded(model) => {
                let shards = model
                    .shards
                    .iter()
                    .map(|shard| {
                        OwnedShard::from_bytes(shard.buffer.to_vec())
                            .expect("valid safetensors data")
                    })
                    .collect::<Vec<_>>();
                Self {
                    storage: ModelStorage::Sharded(OwnedShardedModel {
                        tensor_shards: model.tensor_shards.clone(),
                        tensor_names: model.tensor_names.clone(),
                        shards,
                    }),
                }
            }
        }
    }
}

impl ModelTrait for Model {
    type Tensor<'a> = Tensor<'a> where Self: 'a;

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

#[derive(Debug)]
struct MmapSingleModel {
    shard: MmapShard,
    tensor_names: TensorNameList,
}

impl MmapSingleModel {
    fn from_shard(shard: MmapShard) -> Self {
        let mut tensor_names: Vec<Arc<str>> =
            shard.tensors.names().into_iter().map(Into::into).collect();
        tensor_names.sort_unstable();
        Self {
            shard,
            tensor_names: tensor_names.into(),
        }
    }

    fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        self.shard
            .tensors
            .tensor(name)
            .map(Tensor)
            .map_err(ReaderError::from)
    }

    fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }
}

#[derive(Debug)]
struct MmapShardedModel {
    shards: Vec<MmapShard>,
    tensor_shards: TensorShardMap,
    tensor_names: TensorNameList,
}

impl MmapShardedModel {
    fn from_shards(shards: Vec<MmapShard>) -> ReaderResult<Self> {
        let (tensor_shards, tensor_names) = build_tensor_index(shards.iter().map(|s| &s.tensors))?;
        Ok(Self {
            shards,
            tensor_shards,
            tensor_names,
        })
    }

    fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        let shard_idx =
            self.tensor_shards
                .get(name)
                .copied()
                .ok_or_else(|| ReaderError::TensorNotFound {
                    name: name.to_owned(),
                })?;
        self.shards[shard_idx]
            .tensors
            .tensor(name)
            .map(Tensor)
            .map_err(ReaderError::from)
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

#[derive(Debug)]
enum MmapStorage {
    Single(MmapSingleModel),
    Sharded(MmapShardedModel),
}

/// SafeTensors reader with mmap-backed, lazy storage.
#[derive(Debug)]
pub struct MmapModel {
    storage: MmapStorage,
}

impl MmapModel {
    pub fn open(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let shard_paths = normalize_directory(path)?;
        if shard_paths.len() == 1 {
            return Ok(Self {
                storage: MmapStorage::Single(MmapSingleModel::from_shard(load_mmap(
                    &shard_paths[0],
                )?)),
            });
        }

        let shards: ReaderResult<Vec<_>> = shard_paths
            .into_par_iter()
            .map(|shard_path| load_mmap(&shard_path))
            .collect();
        Ok(Self {
            storage: MmapStorage::Sharded(MmapShardedModel::from_shards(shards?)?),
        })
    }

    #[inline]
    pub fn tensor(&self, name: &str) -> ReaderResult<Tensor<'static>> {
        match &self.storage {
            MmapStorage::Single(model) => model.tensor(name),
            MmapStorage::Sharded(model) => model.tensor(name),
        }
    }

    #[inline]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        match &self.storage {
            MmapStorage::Single(model) => model.tensor_names(),
            MmapStorage::Sharded(model) => model.tensor_names(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match &self.storage {
            MmapStorage::Single(model) => model.tensor_names().len(),
            MmapStorage::Sharded(model) => model.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        match &self.storage {
            MmapStorage::Single(model) => model.tensor_names().is_empty(),
            MmapStorage::Sharded(model) => model.is_empty(),
        }
    }

    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        match &self.storage {
            MmapStorage::Single(model) => model.tensor(name).is_ok(),
            MmapStorage::Sharded(model) => model.contains(name),
        }
    }
}

impl ModelTrait for MmapModel {
    type Tensor<'a> = Tensor<'a> where Self: 'a;

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

impl<'a> TensorView for Tensor<'a> {
    fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    fn dtype(&self) -> &str {
        dtype_to_str(&self.0.dtype())
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
            max_shard_bytes: 2 * 1024 * 1024 * 1024,
        };

        match choose_load_backend(&stats) {
            LoadBackend::Sync => {}
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => {}
            LoadBackend::TokioAsync => panic!("expected Sync or IoUring, got TokioAsync"),
        }
    }

    #[test]
    fn choose_sync_for_many_small_shards() {
        let stats = LoadStats {
            shard_count: 16,
            total_bytes: 256 * 1024 * 1024,
            max_shard_bytes: 32 * 1024 * 1024,
        };

        match choose_load_backend(&stats) {
            LoadBackend::Sync => {}
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => panic!("expected Sync, got IoUring"),
            LoadBackend::TokioAsync => panic!("expected Sync, got TokioAsync"),
        }
    }
}
