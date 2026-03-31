//! ServerlessLLM format model.
//!
//! Provides both eager (owned) and lazy (mmap-backed) model loading.

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::Model as ModelTrait;
use futures::future::try_join_all;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::index::{Index, PartitionPlan};
use super::tensor::{Tensor, TensorMmap};

// ============================================================================
// Eager (owned) model
// ============================================================================

#[derive(Debug)]
struct LoadPlan {
    partitions: Vec<PartitionRead>,
    index: Arc<Index>,
}

#[derive(Debug)]
struct PartitionRead {
    partition_id: usize,
    path: PathBuf,
    size: u64,
}

impl LoadPlan {
    fn compile(index: &Index, base_path: &Path) -> Self {
        let base_path_str = base_path.to_string_lossy();
        let partitions: Vec<PartitionRead> = index
            .partition_ids()
            .iter()
            .filter_map(|partition_id| index.partition(*partition_id))
            .map(|plan: &PartitionPlan| PartitionRead {
                partition_id: plan.partition_id,
                path: PathBuf::from(format!("{}_{}", base_path_str, plan.partition_id)),
                size: plan.max_required_size,
            })
            .collect();
        LoadPlan {
            partitions,
            index: Arc::new(index.clone()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LoadStats {
    partition_count: usize,
    tensor_count: usize,
    total_bytes: u64,
    max_partition_bytes: u64,
}

const SYNC_BASE_COST_NS: f64 = 220_000.0;
const ASYNC_BASE_COST_NS: f64 = 700_000.0;
const SYNC_PER_PARTITION_COST_NS: f64 = 45_000.0;
const ASYNC_PER_PARTITION_COST_NS: f64 = 105_000.0;
const SYNC_PER_TENSOR_COST_NS: f64 = 2_000.0;
const ASYNC_PER_TENSOR_COST_NS: f64 = 3_500.0;
const ASYNC_OVERHEAD_PER_BYTE: f64 = 0.15;
const THROUGHPUT_BPS: f64 = 7.5 * 1024.0 * 1024.0 * 1024.0;
const PARALLELISM_TARGET_BYTES: f64 = 128.0 * 1024.0 * 1024.0;

#[cfg(target_os = "linux")]
const IO_URING_BASE_COST_NS: f64 = 500_000.0;
#[cfg(target_os = "linux")]
const IO_URING_THROUGHPUT_BPS: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;
#[cfg(target_os = "linux")]
const IO_URING_PER_PARTITION_COST_NS: f64 = 50_000.0;
#[cfg(target_os = "linux")]
const IO_URING_PER_TENSOR_COST_NS: f64 = 5_000.0;

fn bytes_to_ns(bytes: u64, throughput_bps: f64) -> f64 {
    (bytes as f64 / throughput_bps) * 1_000_000_000.0
}

fn load_stats(index: &Index) -> LoadStats {
    let mut total_bytes = 0u64;
    let mut max_partition_bytes = 0u64;
    for plan in index.partitions().values() {
        total_bytes = total_bytes.saturating_add(plan.max_required_size);
        max_partition_bytes = max_partition_bytes.max(plan.max_required_size);
    }
    LoadStats {
        partition_count: index.partition_ids().len(),
        tensor_count: index.len(),
        total_bytes,
        max_partition_bytes,
    }
}

fn effective_async_parallelism(stats: &LoadStats) -> f64 {
    if stats.partition_count <= 1 {
        return 1.0;
    }
    let avg = stats.total_bytes.max(1) as f64 / stats.partition_count as f64;
    let max = stats.max_partition_bytes.max(1) as f64;
    let count_factor = (stats.partition_count as f64).ln_1p();
    let size_factor = (PARALLELISM_TARGET_BYTES / avg).sqrt().clamp(0.5, 4.0);
    let skew_factor = (max / avg).sqrt().clamp(1.0, 4.0);
    (1.0 + count_factor * size_factor / skew_factor).clamp(1.0, stats.partition_count as f64)
}

fn estimate_sync_cost(stats: &LoadStats) -> f64 {
    SYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS)
        + SYNC_PER_PARTITION_COST_NS * stats.partition_count as f64
        + SYNC_PER_TENSOR_COST_NS * stats.tensor_count as f64
}

fn estimate_async_cost(stats: &LoadStats) -> f64 {
    let parallelism = effective_async_parallelism(stats);
    ASYNC_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, THROUGHPUT_BPS) / parallelism
        + ASYNC_PER_PARTITION_COST_NS * stats.partition_count as f64
        + ASYNC_PER_TENSOR_COST_NS * stats.tensor_count as f64
        + ASYNC_OVERHEAD_PER_BYTE * stats.max_partition_bytes as f64
}

#[cfg(target_os = "linux")]
fn estimate_io_uring_cost(stats: &LoadStats) -> f64 {
    IO_URING_BASE_COST_NS
        + bytes_to_ns(stats.total_bytes, IO_URING_THROUGHPUT_BPS)
        + IO_URING_PER_PARTITION_COST_NS * stats.partition_count as f64
        + IO_URING_PER_TENSOR_COST_NS * stats.tensor_count as f64
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
        if stats.partition_count <= 1 {
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
        if stats.partition_count <= 1 {
            return LoadBackend::Sync;
        }
        if estimate_async_cost(stats) < estimate_sync_cost(stats) {
            LoadBackend::TokioAsync
        } else {
            LoadBackend::Sync
        }
    }
}

type Tensors = HashMap<Arc<str>, Tensor>;

fn execute_load_plan_sync(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }
    let index = &plan.index;
    for read in &plan.partitions {
        let metadata = std::fs::metadata(&read.path).map_err(|_| ReaderError::PartitionNotFound {
            partition_id: read.partition_id,
            path: read.path.to_string_lossy().to_string(),
        })?;
        if metadata.len() < read.size {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: metadata.len(),
                required: read.size,
            });
        }
    }
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let mut reader = backends::SyncReader::new();
        let data = reader.load(&read.path).map_err(ReaderError::from)?;
        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }
        let backing: Arc<[u8]> = data.into_shared();
        let mut tensors = HashMap::with_capacity(index.len());
        for name in index.tensor_names().iter() {
            let desc = index.get(name.as_ref()).unwrap();
            tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)));
        }
        return Ok(tensors);
    }
    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();
    let mut reader = backends::SyncReader::new();
    let results = reader.load_range_batch(&requests).map_err(ReaderError::from)?;
    let max_partition_id = plan.partitions.iter().map(|r| r.partition_id).max().unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (read, result) in plan.partitions.iter().zip(results) {
        let (buf, _, _) = result;
        if buf.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: buf.len() as u64,
                required: read.size,
            });
        }
        partition_buffers[read.partition_id] = Some(buf);
    }
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)));
    }
    Ok(tensors)
}

#[cfg(target_os = "linux")]
fn execute_load_plan_io_uring(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }
    let index = &plan.index;
    for read in &plan.partitions {
        let metadata = std::fs::metadata(&read.path).map_err(|_| ReaderError::PartitionNotFound {
            partition_id: read.partition_id,
            path: read.path.to_string_lossy().to_string(),
        })?;
        if metadata.len() < read.size {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: metadata.len(),
                required: read.size,
            });
        }
    }

    let mut reader = backends::io_uring::Reader::new()?;

    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let data = reader.load(&read.path).map_err(ReaderError::from)?;
        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }
        let backing: Arc<[u8]> = data.into_shared();
        let mut tensors = HashMap::with_capacity(index.len());
        for name in index.tensor_names().iter() {
            let desc = index.get(name.as_ref()).unwrap();
            tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)));
        }
        return Ok(tensors);
    }

    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();
    let results = reader.load_range_batch(&requests).map_err(ReaderError::from)?;
    let max_partition_id = plan.partitions.iter().map(|r| r.partition_id).max().unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (read, (buf, _, _)) in plan.partitions.iter().zip(results) {
        if buf.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: buf.len() as u64,
                required: read.size,
            });
        }
        partition_buffers[read.partition_id] = Some(buf);
    }
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)));
    }
    Ok(tensors)
}

async fn execute_load_plan_async(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }
    let index = &plan.index;
    let validations: Vec<_> = plan
        .partitions
        .iter()
        .map(|read| async {
            let metadata = tokio::fs::metadata(&read.path).await.map_err(|_| {
                ReaderError::PartitionNotFound {
                    partition_id: read.partition_id,
                    path: read.path.to_string_lossy().to_string(),
                }
            })?;
            if metadata.len() < read.size {
                return Err(ReaderError::PartitionTooSmall {
                    path: read.path.to_string_lossy().to_string(),
                    actual: metadata.len(),
                    required: read.size,
                });
            }
            Ok(())
        })
        .collect();
    try_join_all(validations).await?;
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let mut reader = backends::AsyncReader::new();
        let data = reader.load(&read.path).await.map_err(ReaderError::from)?;
        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }
        let backing: Arc<[u8]> = data.into_shared();
        let mut tensors = HashMap::with_capacity(index.len());
        for name in index.tensor_names().iter() {
            let desc = index.get(name.as_ref()).unwrap();
            tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)));
        }
        return Ok(tensors);
    }
    let load_futures: Vec<_> = plan
        .partitions
        .iter()
        .map(|read| async move {
            let mut reader = backends::AsyncReader::new();
            let data = reader.load(&read.path).await.map_err(ReaderError::from)?;
            if data.len() < read.size as usize {
                return Err(ReaderError::PartitionTooSmall {
                    path: read.path.to_string_lossy().to_string(),
                    actual: data.len() as u64,
                    required: read.size,
                });
            }
            let owned: Arc<[u8]> = data.into_shared();
            Ok((read.partition_id, owned))
        })
        .collect();
    let results: Vec<(usize, Arc<[u8]>)> = futures::future::join_all(load_futures)
        .await
        .into_iter()
        .collect::<ReaderResult<Vec<_>>>()?;
    let max_partition_id = plan.partitions.iter().map(|r| r.partition_id).max().unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (partition_id, buf) in results {
        partition_buffers[partition_id] = Some(buf);
    }
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(name.clone(), Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)));
    }
    Ok(tensors)
}

/// ServerlessLLM model with all tensors loaded into memory (eager loading).
#[derive(Debug, Clone)]
pub struct Model {
    tensors: HashMap<Arc<str>, Tensor>,
    tensor_names: Arc<[Arc<str>]>,
}

impl Model {
    fn compile_load_plan(directory: impl AsRef<Path>) -> ReaderResult<(Index, LoadPlan)> {
        let dir_path = directory.as_ref();
        let index = Index::load_sync(dir_path.join("tensor_index.json"))?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        Ok((index, plan))
    }

    pub async fn load(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        match choose_load_backend(&load_stats(&index)) {
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => {
                let tensors = execute_load_plan_io_uring(&plan)?;
                Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
            }
            LoadBackend::TokioAsync => {
                let tensors = execute_load_plan_async(&plan).await?;
                Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
            }
            LoadBackend::Sync => {
                let tensors = execute_load_plan_sync(&plan)?;
                Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub fn load_io_uring(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        let tensors = execute_load_plan_io_uring(&plan)?;
        Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
    }

    pub async fn load_async(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load(dir_path.join("tensor_index.json")).await?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        let tensors = execute_load_plan_async(&plan).await?;
        Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
    }

    pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        let tensors = execute_load_plan_sync(&plan)?;
        Ok(Self { tensors, tensor_names: index.tensor_names().to_vec().into() })
    }

    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

impl<'a> IntoIterator for &'a Model {
    type Item = (&'a Arc<str>, &'a Tensor);
    type IntoIter = std::collections::hash_map::Iter<'a, Arc<str>, Tensor>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl ModelTrait for Model {
    type Tensor<'a> = &'a Tensor where Self: 'a;

    #[inline]
    fn len(&self) -> usize {
        self.tensors.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[inline]
    fn tensor_names(&self) -> &[Arc<str>] {
        Model::tensor_names(self)
    }

    #[inline]
    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        self.tensors.get(name)
    }
}

// ============================================================================
// Lazy (mmap) model
// ============================================================================

/// ServerlessLLM model with memory-mapped partition files (lazy loading).
#[derive(Debug)]
pub struct MmapModel {
    index: Index,
    partitions: HashMap<usize, backends::mmap::Mmap>,
}

impl MmapModel {
    pub fn open(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load_sync(dir_path.join("tensor_index.json"))?;
        let partition_ids = index.partition_ids();
        let partitions: Result<HashMap<usize, backends::mmap::Mmap>, ReaderError> = partition_ids
            .par_iter()
            .map(|&partition_id| {
                let partition_path = format!(
                    "{}_{}",
                    dir_path.join("tensor.data").display(),
                    partition_id
                );
                let mmap = backends::mmap::map(&partition_path)?;
                Ok((partition_id, mmap))
            })
            .collect();
        Ok(Self {
            index,
            partitions: partitions?,
        })
    }

    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<TensorMmap> {
        let desc = self.index.get(name)?;
        let mmap = self.partitions.get(&desc.partition_id)?;
        let start = usize::try_from(desc.offset).ok()?;
        let end = start.checked_add(desc.size)?;
        if end > mmap.len() {
            return None;
        }
        let tensor_mmap = backends::mmap::Mmap {
            inner: Arc::clone(&mmap.inner),
            start: mmap.start + start,
            len: desc.size,
        };
        Some(TensorMmap::new(tensor_mmap, Arc::clone(desc)))
    }

    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        self.index.tensor_names()
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.index.tensor_names().iter().map(|s| s.as_ref())
    }
}

impl ModelTrait for MmapModel {
    type Tensor<'a> = TensorMmap where Self: 'a;

    #[inline]
    fn len(&self) -> usize {
        self.index.len()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.index.contains(name)
    }

    #[inline]
    fn tensor_names(&self) -> &[Arc<str>] {
        MmapModel::tensor_names(self)
    }

    #[inline]
    fn tensor(&self, name: &str) -> Option<Self::Tensor<'_>> {
        MmapModel::tensor(self, name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_len_and_is_empty() {
        let model = Model {
            tensors: HashMap::new(),
            tensor_names: Arc::new([]),
        };
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn choose_sync_for_single_large_partition() {
        let stats = LoadStats {
            partition_count: 1,
            tensor_count: 1,
            total_bytes: 2 * 1024 * 1024 * 1024,
            max_partition_bytes: 2 * 1024 * 1024 * 1024,
        };
        match choose_load_backend(&stats) {
            LoadBackend::Sync => {}
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => {}
            LoadBackend::TokioAsync => panic!("expected Sync or IoUring, got TokioAsync"),
        }
    }

    #[test]
    fn choose_async_for_many_small_partitions() {
        let stats = LoadStats {
            partition_count: 16,
            tensor_count: 1024,
            total_bytes: 256 * 1024 * 1024,
            max_partition_bytes: 32 * 1024 * 1024,
        };
        match choose_load_backend(&stats) {
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => {}
            LoadBackend::TokioAsync => {}
            LoadBackend::Sync => panic!("expected TokioAsync or IoUring, got Sync"),
        }
    }

    #[test]
    fn choose_sync_for_low_fanout_partitions() {
        let stats = LoadStats {
            partition_count: 2,
            tensor_count: 160,
            total_bytes: 524 * 1024 * 1024,
            max_partition_bytes: 334 * 1024 * 1024,
        };
        match choose_load_backend(&stats) {
            LoadBackend::Sync => {}
            #[cfg(target_os = "linux")]
            LoadBackend::IoUring => {}
            LoadBackend::TokioAsync => panic!("expected Sync or IoUring, got TokioAsync"),
        }
    }

    #[test]
    fn mmap_model_empty() {
        let model = MmapModel {
            index: Index::new(),
            partitions: HashMap::new(),
        };
        assert!(model.is_empty());
    }
}
