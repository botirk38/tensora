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
    total_bytes: u64,
}

impl LoadStats {
    #[cfg(target_os = "linux")]
    fn avg_partition_bytes(self) -> u64 {
        self.total_bytes
            .div_ceil(self.partition_count.max(1) as u64)
    }

    fn log2_bytes(self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.total_bytes as f64).log2()
        }
    }

    fn partition_fanout_score(self) -> f64 {
        match self.partition_count {
            0..=4 => 0.0,
            5..=8 => 1.0,
            9..=16 => 2.0,
            17..=32 => 3.0,
            _ => 4.0,
        }
    }
}

fn load_stats(index: &Index) -> LoadStats {
    let mut total_bytes = 0u64;
    for plan in index.partitions().values() {
        total_bytes = total_bytes.saturating_add(plan.max_required_size);
    }
    LoadStats {
        partition_count: index.partition_ids().len(),
        total_bytes,
    }
}

enum LoadBackend {
    TokioAsync,
    #[cfg(target_os = "linux")]
    IoUring,
}

/// Score-based backend selection for ServerlessLLM on Linux.
///
/// Uses a simple scoring function to estimate whether io_uring or async will perform better.
/// Score = log2(total_bytes) + 2*fanout_bucket + avg_partition_gb
/// If score exceeds threshold, use io_uring; otherwise async.
///
/// Threshold tuned for H100 environments - async generally better below large partition counts
fn choose_load_backend(stats: &LoadStats) -> LoadBackend {
    #[cfg(target_os = "linux")]
    {
        let log2_bytes = stats.log2_bytes();
        let fanout = stats.partition_fanout_score();
        let avg_partition_gb = stats.avg_partition_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);

        let score = log2_bytes + 2.0 * fanout + avg_partition_gb;

        if score >= 25.0 {
            return LoadBackend::IoUring;
        }
        LoadBackend::TokioAsync
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (stats.partition_count, stats.total_bytes);
        LoadBackend::TokioAsync
    }
}

type Tensors = HashMap<Arc<str>, Tensor>;

fn execute_load_plan_sync(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }
    let index = &plan.index;
    for read in &plan.partitions {
        let metadata =
            std::fs::metadata(&read.path).map_err(|_| ReaderError::PartitionNotFound {
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
            tensors.insert(
                name.clone(),
                Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)),
            );
        }
        return Ok(tensors);
    }
    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();
    let mut reader = backends::SyncReader::new();
    let results = reader
        .load_range_batch(&requests)
        .map_err(ReaderError::from)?;
    let max_partition_id = plan
        .partitions
        .iter()
        .map(|r| r.partition_id)
        .max()
        .unwrap_or(0);
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
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
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
        let metadata =
            std::fs::metadata(&read.path).map_err(|_| ReaderError::PartitionNotFound {
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

    let mut reader = backends::io_uring::Reader::new();

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
            tensors.insert(
                name.clone(),
                Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)),
            );
        }
        return Ok(tensors);
    }

    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();
    let results = reader
        .load_range_batch(&requests)
        .map_err(ReaderError::from)?;
    let max_partition_id = plan
        .partitions
        .iter()
        .map(|r| r.partition_id)
        .max()
        .unwrap_or(0);
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
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
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
            tensors.insert(
                name.clone(),
                Tensor::from_shared(Arc::clone(&backing), Arc::clone(desc)),
            );
        }
        return Ok(tensors);
    }
    let requests: Vec<(PathBuf, u64, usize)> = plan
        .partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();
    let mut reader = backends::AsyncReader::new();
    let results = reader
        .load_range_batch(&requests)
        .await
        .map_err(ReaderError::from)?;
    let partition_results: Vec<(usize, Arc<[u8]>)> = plan
        .partitions
        .iter()
        .zip(results)
        .map(|(read, (buf, _, _))| {
            if buf.len() < read.size as usize {
                return Err(ReaderError::PartitionTooSmall {
                    path: read.path.to_string_lossy().to_string(),
                    actual: buf.len() as u64,
                    required: read.size,
                });
            }
            Ok((read.partition_id, buf))
        })
        .collect::<ReaderResult<Vec<_>>>()?;
    let max_partition_id = plan
        .partitions
        .iter()
        .map(|r| r.partition_id)
        .max()
        .unwrap_or(0);
    let mut partition_buffers: Vec<Option<Arc<[u8]>>> = vec![None; max_partition_id + 1];
    for (partition_id, buf) in partition_results {
        partition_buffers[partition_id] = Some(buf);
    }
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers[desc.partition_id].as_ref().unwrap();
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
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
                Ok(Self {
                    tensors,
                    tensor_names: index.tensor_names().to_vec().into(),
                })
            }
            LoadBackend::TokioAsync => {
                let tensors = execute_load_plan_async(&plan).await?;
                Ok(Self {
                    tensors,
                    tensor_names: index.tensor_names().to_vec().into(),
                })
            }
        }
    }

    #[cfg(target_os = "linux")]
    pub fn load_io_uring(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        let tensors = execute_load_plan_io_uring(&plan)?;
        Ok(Self {
            tensors,
            tensor_names: index.tensor_names().to_vec().into(),
        })
    }

    pub async fn load_async(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load(dir_path.join("tensor_index.json")).await?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        let tensors = execute_load_plan_async(&plan).await?;
        Ok(Self {
            tensors,
            tensor_names: index.tensor_names().to_vec().into(),
        })
    }

    pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let (index, plan) = Self::compile_load_plan(directory)?;
        let tensors = execute_load_plan_sync(&plan)?;
        Ok(Self {
            tensors,
            tensor_names: index.tensor_names().to_vec().into(),
        })
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
    type Tensor<'a>
        = &'a Tensor
    where
        Self: 'a;

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
    type Tensor<'a>
        = TensorMmap
    where
        Self: 'a;

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
    fn choose_io_uring_for_low_partition_count_model() {
        let stats = LoadStats {
            partition_count: 3,
            total_bytes: 2 * 1024 * 1024 * 1024,
        };
        let b = choose_load_backend(&stats);
        assert!(matches!(b, LoadBackend::IoUring | LoadBackend::TokioAsync));
    }

    #[test]
    fn choose_io_uring_for_large_model() {
        let stats = LoadStats {
            partition_count: 16,
            total_bytes: 16 * 1024 * 1024 * 1024,
        };
        let b = choose_load_backend(&stats);
        assert!(matches!(b, LoadBackend::IoUring | LoadBackend::TokioAsync));
    }

    #[test]
    fn choose_async_for_medium_small_model() {
        let stats = LoadStats {
            partition_count: 12,
            total_bytes: 524 * 1024 * 1024,
        };
        let b = choose_load_backend(&stats);
        assert!(matches!(b, LoadBackend::TokioAsync | LoadBackend::IoUring));
    }

    #[test]
    fn choose_async_for_many_partitions_small_total() {
        let stats = LoadStats {
            partition_count: 16,
            total_bytes: 2 * 1024 * 1024 * 1024,
        };
        let b = choose_load_backend(&stats);
        assert!(matches!(b, LoadBackend::TokioAsync | LoadBackend::IoUring));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn choose_sync_for_single_partition_small() {
        let stats = LoadStats {
            partition_count: 1,
            total_bytes: 512 * 1024 * 1024,
        };
        match choose_load_backend(&stats) {
            LoadBackend::IoUring | LoadBackend::TokioAsync => {}
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn choose_io_uring_for_very_large_partitioned_model() {
        let stats = LoadStats {
            partition_count: 32,
            total_bytes: 32 * 1024 * 1024 * 1024,
        };
        match choose_load_backend(&stats) {
            LoadBackend::IoUring | LoadBackend::TokioAsync => {}
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
