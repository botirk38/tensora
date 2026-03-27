//! Eager loading for ServerlessLLM format (owned buffers).

use crate::backends;
use crate::formats::error::{ReaderError, ReaderResult};
use crate::formats::traits::TensorMetadata;
use futures::future;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::index::Index;
use super::tensor::Tensor;
use super::helpers;

/// ServerlessLLM model with all tensors loaded into memory (eager loading).
#[derive(Debug, Clone)]
pub struct Model {
    tensors: HashMap<String, Tensor>,
}

impl Model {
    /// Loads a ServerlessLLM model from directory asynchronously with eager loading.
    pub async fn load(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = Index::load(index_path.as_path()).await?;
        let tensor_names: Vec<&str> = index.tensor_names();

        let loaded = load_tensors_batch_zero_copy_async(&index, &data_path, &tensor_names).await?;

        let tensors = loaded
            .into_iter()
            .map(|(name, (buf, offset, len))| {
                let entry = index.get(&name).unwrap().clone();
                (name, Tensor::new(buf, offset, len, entry))
            })
            .collect();

        Ok(Self { tensors })
    }

    /// Loads a ServerlessLLM model from directory synchronously with eager loading.
    pub fn load_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = Index::load_sync(index_path.as_path())?;
        let tensor_names: Vec<&str> = index.tensor_names();

        let tensor_data = load_tensors_batch_sync(&index, &data_path, &tensor_names)?;

        let tensors = tensor_data
            .into_iter()
            .map(|(name, data)| {
                let entry = index.get(&name).unwrap().clone();
                (name, Tensor::from_owned(data, entry))
            })
            .collect();

        Ok(Self { tensors })
    }

    /// Loads a ServerlessLLM model with partition-parallel loading (async).
    pub async fn load_parallel(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = Index::load_sync(&index_path)?;
        let partition_ids = index.partition_ids();

        if partition_ids.is_empty() {
            return Ok(Self {
                tensors: HashMap::new(),
            });
        }

        let base_path_str = data_path.to_string_lossy().to_string();
        let partition_paths: Vec<(usize, PathBuf)> = partition_ids
            .iter()
            .map(|&id| (id, PathBuf::from(format!("{}_{}", base_path_str, id))))
            .collect();

        const GLOBAL_QUEUE_BUDGET: usize = 64;
        let chunks_per_partition = GLOBAL_QUEUE_BUDGET.div_ceil(partition_paths.len());
        let partition_data: Vec<(usize, Vec<u8>)> = future::try_join_all(
            partition_paths
                .iter()
                .map(|(partition_id, path)| async move {
                    let data = backends::async_backend()
                        .load_parallel(path.as_path(), chunks_per_partition)
                        .await?;
                    Ok::<_, ReaderError>((*partition_id, data))
                }),
        )
        .await?;

        let partitions: HashMap<usize, Vec<u8>> = partition_data.into_iter().collect();

        let tensors: HashMap<String, Tensor> = index
            .iter()
            .map(|(name, entry)| {
                let partition_buf = partitions.get(&entry.partition_id).ok_or_else(|| {
                    ReaderError::PartitionNotFound {
                        partition_id: entry.partition_id,
                        path: format!("{}_{}", base_path_str, entry.partition_id),
                    }
                })?;

                let data = helpers::slice_tensor_from_partition(name.clone(), partition_buf, entry)?
                    .into_data();
                let entry_owned = entry.clone();
                Ok((name.clone(), Tensor::from_owned(data, entry_owned)))
            })
            .collect::<ReaderResult<HashMap<_, _>>>()?;

        Ok(Self { tensors })
    }

    /// Loads a ServerlessLLM model with partition-parallel loading (sync).
    pub fn load_parallel_sync(directory: impl AsRef<Path>) -> ReaderResult<Self> {
        let dir_path = directory.as_ref();
        let index_path = dir_path.join("tensor_index.json");
        let data_path = dir_path.join("tensor.data");

        let index = Index::load_sync(&index_path)?;
        let partition_ids = index.partition_ids();

        if partition_ids.is_empty() {
            return Ok(Self {
                tensors: HashMap::new(),
            });
        }

        let base_path_str = data_path.to_string_lossy().to_string();
        let partition_paths: Vec<(usize, String)> = partition_ids
            .iter()
            .map(|&id| (id, format!("{}_{}", base_path_str, id)))
            .collect();

        const GLOBAL_QUEUE_BUDGET: usize = 64;
        let chunks_per_partition = GLOBAL_QUEUE_BUDGET.div_ceil(partition_paths.len());
        let partition_data: Vec<(usize, Vec<u8>)> = partition_paths
            .par_iter()
            .map(|(partition_id, path)| {
                let data = backends::sync_backend()
                    .load_parallel(Path::new(path), chunks_per_partition)?;
                Ok::<_, std::io::Error>((*partition_id, data))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ReaderError::from)?;

        let partitions: HashMap<usize, Vec<u8>> = partition_data.into_iter().collect();

        let tensors: HashMap<String, Tensor> = index
            .iter()
            .map(|(name, entry)| {
                let partition_buf = partitions.get(&entry.partition_id).ok_or_else(|| {
                    ReaderError::PartitionNotFound {
                        partition_id: entry.partition_id,
                        path: format!("{}_{}", base_path_str, entry.partition_id),
                    }
                })?;

                let data = helpers::slice_tensor_from_partition(name.clone(), partition_buf, entry)?
                    .into_data();
                let entry_owned = entry.clone();
                Ok((name.clone(), Tensor::from_owned(data, entry_owned)))
            })
            .collect::<ReaderResult<HashMap<_, _>>>()?;

        Ok(Self { tensors })
    }

    /// Returns a reference to the tensor with the given name.
    #[inline]
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Returns tensor names.
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

    /// Returns an iterator over all tensor names and tensors.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }

    /// Consumes the model and returns the underlying tensor map.
    #[inline]
    #[must_use]
    pub fn into_tensors(self) -> HashMap<String, Tensor> {
        self.tensors
    }
}

impl<'a> IntoIterator for &'a Model {
    type Item = (&'a String, &'a Tensor);
    type IntoIter = std::collections::hash_map::Iter<'a, String, Tensor>;

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
    fn tensor_names(&self) -> Vec<&str> {
        self.tensors
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }
}



// ============================================================================
// Internal batch loading helpers
// ============================================================================

async fn load_tensors_batch_zero_copy_async(
    index: &Index,
    base_path: &Path,
    tensor_names: &[&str],
) -> ReaderResult<HashMap<String, (Vec<u8>, usize, usize)>> {
    use std::collections::HashMap as StdHashMap;

    let partition_requests = helpers::group_requests_by_partition(index, tensor_names)?;

    let base_path_str = base_path.to_string_lossy();
    let partition_paths: StdHashMap<usize, String> = partition_requests
        .keys()
        .map(|&partition_id| (partition_id, format!("{}_{}", base_path_str, partition_id)))
        .collect();

    // Stat all partitions concurrently
    let partition_sizes_vec: Vec<(usize, u64)> = future::try_join_all(partition_paths.iter().map(
        |(partition_id, path): (&usize, &String)| async move {
            let meta = tokio::fs::metadata(path).await.map_err(|_| {
                ReaderError::PartitionNotFound {
                    partition_id: *partition_id,
                    path: path.clone(),
                }
            })?;
            ReaderResult::Ok((*partition_id, meta.len()))
        },
    ))
    .await?;

    let partition_sizes: StdHashMap<usize, u64> = partition_sizes_vec.into_iter().collect();

    // Validate all tensors
    let mut batch_requests: Vec<backends::BatchRequest> = Vec::with_capacity(tensor_names.len());
    let mut names = Vec::with_capacity(tensor_names.len());

    for &name in tensor_names {
        let entry = index.get(name).unwrap();
        let required_size = helpers::required_size(entry, name)?;

        let partition_path = partition_paths.get(&entry.partition_id).ok_or_else(|| {
            ReaderError::PartitionNotFound {
                partition_id: entry.partition_id,
                path: format!("{}_{}", base_path_str, entry.partition_id),
            }
        })?;

        let actual_size = *partition_sizes.get(&entry.partition_id).ok_or_else(|| {
            ReaderError::PartitionNotFound {
                partition_id: entry.partition_id,
                path: partition_path.clone(),
            }
        })?;

        if actual_size < required_size {
            return Err(ReaderError::PartitionTooSmall {
                path: partition_path.clone(),
                actual: actual_size,
                required: required_size,
            });
        }

        let len = usize::try_from(entry.size).map_err(|_| ReaderError::SizeTooLarge {
            name: name.to_string(),
            size: entry.size,
        })?;

        batch_requests.push((PathBuf::from(partition_path), entry.offset, len));
        names.push(name.to_string());
    }

    let batch_results = backends::async_backend()
        .load_batch(&batch_requests)
        .await
        .map_err(ReaderError::from)?;

    Ok(names.into_iter().zip(batch_results).collect())
}

fn load_tensors_batch_sync(
    index: &Index,
    base_path: &Path,
    tensor_names: &[&str],
) -> ReaderResult<HashMap<String, Vec<u8>>> {
    use std::collections::HashMap as StdHashMap;

    let partition_requests = helpers::group_requests_by_partition(index, tensor_names)?;

    let base_path_str = base_path.to_string_lossy();
    let partition_paths: StdHashMap<usize, String> = partition_requests
        .keys()
        .map(|&partition_id| (partition_id, format!("{}_{}", base_path_str, partition_id)))
        .collect();

     // Parallel validation of all partitions
     let partition_validations: StdHashMap<usize, (String, u64)> = tensor_names
         .iter()
         .try_fold(StdHashMap::new(), |mut map: StdHashMap<usize, (String, u64)>, &name| {
             let entry = index.get(name).unwrap();
             let required_size = helpers::required_size(entry, name)?;
             let path = partition_paths
                 .get(&entry.partition_id)
                 .ok_or_else(|| ReaderError::PartitionNotFound {
                     partition_id: entry.partition_id,
                     path: format!("{}_{}", base_path_str, entry.partition_id),
                 })?
                 .clone();
             map.entry(entry.partition_id)
                 .and_modify(|(_, max_size): &mut (String, u64)| *max_size = (*max_size).max(required_size))
                 .or_insert((path, required_size));
             Ok::<StdHashMap<usize, (String, u64)>, ReaderError>(map)
         })?;

     let validation_results: Vec<Result<(usize, u64), ReaderError>> = partition_validations
         .par_iter()
         .map(|(&partition_id, (path, required_size))| {
             let metadata = std::fs::metadata(path.as_str()).map_err(|_| ReaderError::PartitionNotFound {
                 partition_id,
                 path: path.clone(),
             })?;
            let actual_size = metadata.len();
            if actual_size < *required_size {
                return Err(ReaderError::PartitionTooSmall {
                    path: path.clone(),
                    actual: actual_size,
                    required: *required_size,
                });
            }
            Ok((partition_id, actual_size))
        })
        .collect();

    for result in validation_results {
        result?;
    }

    // Build batch requests
    let mut batch_requests: Vec<backends::BatchRequest> = Vec::with_capacity(tensor_names.len());
    let mut names = Vec::with_capacity(tensor_names.len());

    for &name in tensor_names {
        let entry = index.get(name).unwrap();
        let partition_path = partition_paths.get(&entry.partition_id).ok_or_else(|| {
            ReaderError::PartitionNotFound {
                partition_id: entry.partition_id,
                path: format!("{}_{}", base_path_str, entry.partition_id),
            }
        })?;

        let len = usize::try_from(entry.size).map_err(|_| ReaderError::SizeTooLarge {
            name: name.to_string(),
            size: entry.size,
        })?;

        batch_requests.push((PathBuf::from(partition_path.as_str()), entry.offset, len));
        names.push(name.to_string());
    }

    let batch_results = backends::sync_backend()
        .load_range_batch(&batch_requests)
        .map_err(ReaderError::from)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_len_and_is_empty() {
        let model = Model {
            tensors: HashMap::new(),
        };
        assert!(model.is_empty());
        assert_eq!(model.len(), 0);
    }
}
