//! Eager loading for ServerlessLLM format (owned buffers).
//!
//! Partition-native loading: one read per partition, shared backing buffers.

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::TensorMetadata;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::index::{Index, PartitionPlan};
use super::tensor::Tensor;

/// Load plan: one read per partition, tensors assembled from shared backing.
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
            .partitions()
            .values()
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

type Tensors = HashMap<Arc<str>, Tensor>;

/// Execute load plan synchronously - partition-native with shared backing.
fn execute_load_plan_sync(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }

    let index = &plan.index;

    // Validate all partitions first
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

    // Fast path: single partition - use direct load, avoid batch overhead
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let data = backends::sync_backend()
            .load(&read.path)
            .map_err(ReaderError::from)?;
        
        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }
        
        let backing: Arc<[u8]> = data.into();

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

    // Multi-partition: parallel batch reads
    let requests: Vec<(PathBuf, u64, usize)> = plan.partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();

    let results = backends::sync_backend()
        .load_range_batch(&requests)
        .map_err(ReaderError::from)?;

    // Build partition buffers without extra copy
    let mut partition_buffers: HashMap<usize, Arc<[u8]>> = HashMap::with_capacity(plan.partitions.len());
    for (read, result) in plan.partitions.iter().zip(results) {
        let (buf, _, _) = result;
        if buf.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: buf.len() as u64,
                required: read.size,
            });
        }
        partition_buffers.insert(read.partition_id, buf);
    }

    // Assemble tensors from shared partition buffers
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers.get(&desc.partition_id).unwrap();
        tensors.insert(
            name.clone(),
            Tensor::from_shared(Arc::clone(backing), Arc::clone(desc)),
        );
    }

    Ok(tensors)
}

/// Execute load plan asynchronously - partition-native with shared backing.
async fn execute_load_plan_async(plan: &LoadPlan) -> ReaderResult<Tensors> {
    if plan.partitions.is_empty() {
        return Ok(HashMap::new());
    }

    let index = &plan.index;

    // Validate all partitions first
    for read in &plan.partitions {
        let metadata = tokio::fs::metadata(&read.path).await.map_err(|_| ReaderError::PartitionNotFound {
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

    // Fast path: single partition - use direct load, avoid batch overhead
    if plan.partitions.len() == 1 {
        let read = &plan.partitions[0];
        let data = backends::async_backend()
            .load(&read.path)
            .await
            .map_err(ReaderError::from)?;
        
        if data.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: data.len() as u64,
                required: read.size,
            });
        }
        
        let backing: Arc<[u8]> = data.into();

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

    // Multi-partition: parallel batch reads
    let requests: Vec<(PathBuf, u64, usize)> = plan.partitions
        .iter()
        .map(|p| (p.path.clone(), 0, p.size as usize))
        .collect();

    // Read all partitions in parallel via batch
    let results = backends::async_backend()
        .load_batch(&requests)
        .await
        .map_err(ReaderError::from)?;

    // Build partition buffers without extra copy
    let mut partition_buffers: HashMap<usize, Arc<[u8]>> = HashMap::with_capacity(plan.partitions.len());
    for (read, result) in plan.partitions.iter().zip(results) {
        let (buf, _, _) = result;
        if buf.len() < read.size as usize {
            return Err(ReaderError::PartitionTooSmall {
                path: read.path.to_string_lossy().to_string(),
                actual: buf.len() as u64,
                required: read.size,
            });
        }
        partition_buffers.insert(read.partition_id, buf);
    }

    // Assemble tensors from shared partition buffers
    let mut tensors = HashMap::with_capacity(index.len());
    for name in index.tensor_names().iter() {
        let desc = index.get(name.as_ref()).unwrap();
        let backing = partition_buffers.get(&desc.partition_id).unwrap();
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
    /// Loads a ServerlessLLM model from directory asynchronously with eager loading.
    pub async fn load(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load(dir_path.join("tensor_index.json")).await?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        let tensors = execute_load_plan_async(&plan).await?;
        let tensor_names = index.tensor_names().to_vec().into();
        Ok(Self { tensors, tensor_names })
    }

    /// Loads a ServerlessLLM model from directory synchronously with eager loading.
    pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index = Index::load_sync(dir_path.join("tensor_index.json"))?;
        let plan = LoadPlan::compile(&index, &dir_path.join("tensor.data"));
        let tensors = execute_load_plan_sync(&plan)?;
        let tensor_names = index.tensor_names().to_vec().into();
        Ok(Self { tensors, tensor_names })
    }

    /// Returns a reference to the tensor with the given name.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Returns tensor names (cached, sorted).
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names
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
}

impl<'a> IntoIterator for &'a Model {
    type Item = (&'a Arc<str>, &'a Tensor);
    type IntoIter = std::collections::hash_map::Iter<'a, Arc<str>, Tensor>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.tensors.iter()
    }
}

impl TensorMetadata for Model {
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
}
