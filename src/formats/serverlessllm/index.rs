//! ServerlessLLM index parsing and metadata.
//!
//! Redesigned for compiled metadata - all partition info computed once at parse time.

use crate::formats::error::{ReaderError, ReaderResult};
use crate::io::AsyncIo;
use crate::io::tokio::Tokio;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use super::ids::PartitionId;

/// Compact tensor descriptor for load planning.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub offset: u64,
    pub size: usize,
    pub shape: Arc<[usize]>,
    pub stride: Arc<[usize]>,
    pub dtype: Arc<str>,
    pub partition_id: PartitionId,
}

/// Precomputed partition metadata for load planning.
#[derive(Debug, Clone)]
pub struct PartitionPlan {
    pub partition_id: PartitionId,
    pub max_required_size: u64,
    pub tensor_names: Arc<[Arc<str>]>,
}

/// Compiled ServerlessLLM index with cached partition metadata.
#[derive(Debug, Clone)]
pub struct Index {
    tensors: HashMap<Arc<str>, Arc<TensorDescriptor>>,
    partition_ids: Arc<[PartitionId]>,
    partitions: HashMap<PartitionId, PartitionPlan>,
    tensor_names_sorted: Arc<[Arc<str>]>,
}

impl Index {
    /// Creates a new empty index.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            partition_ids: Arc::new([]),
            partitions: HashMap::new(),
            tensor_names_sorted: Arc::new([]),
        }
    }

    /// Gets a tensor descriptor by name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Arc<TensorDescriptor>> {
        self.tensors.get(name)
    }

    /// Returns true if a tensor with the given name exists.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
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

    /// Returns tensor names in sorted order (cached).
    #[inline]
    #[must_use]
    pub fn tensor_names(&self) -> &[Arc<str>] {
        &self.tensor_names_sorted
    }

    /// Returns an iterator over tensor names and descriptors.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Arc<TensorDescriptor>)> {
        self.tensors.iter()
    }

    /// Returns sorted partition IDs (cached, computed once at parse time).
    #[inline]
    #[must_use]
    pub fn partition_ids(&self) -> &[PartitionId] {
        &self.partition_ids
    }

    /// Returns partition plans (cached).
    #[inline]
    #[must_use]
    pub fn partitions(&self) -> &HashMap<PartitionId, PartitionPlan> {
        &self.partitions
    }

    /// Returns the partition plan for a given partition ID.
    #[inline]
    #[must_use]
    pub fn partition(&self, id: PartitionId) -> Option<&PartitionPlan> {
        self.partitions.get(&id)
    }

    /// Parse index from raw bytes.
    pub fn from_bytes(data: &[u8]) -> ReaderResult<Self> {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(data)
            .map_err(|err| ReaderError::ServerlessLlm(format!("JSON parse error: {err}")))?;

        let mut tensors = HashMap::with_capacity(raw.len());
        let mut partition_max_sizes: HashMap<PartitionId, u64> = HashMap::new();
        let mut partition_tensors: HashMap<PartitionId, Vec<Arc<str>>> = HashMap::new();

        for (name, value) in raw {
            let name: Arc<str> = name.as_str().into();
            let arr = value.as_array().ok_or_else(|| {
                ReaderError::ServerlessLlm("tensor entry must be an array".to_string())
            })?;

            let arr = arr.as_slice();
            if arr.len() != 6 {
                return Err(ReaderError::ServerlessLlm(format!(
                    "tensor entry must have exactly 6 elements, got {}",
                    arr.len()
                )));
            }

            let offset = parse_u64(&arr[0])?;
            let size = parse_usize(&arr[1])?;
            let size_u64 = size as u64;
            let shape = parse_usize_vec(&arr[2])?;
            let stride = parse_usize_vec(&arr[3])?;
            let dtype = parse_string(&arr[4])?;
            let partition_id = PartitionId::new(parse_usize(&arr[5])?);
            let required =
                offset
                    .checked_add(size_u64)
                    .ok_or_else(|| ReaderError::OffsetOverflow {
                        name: name.to_string(),
                    })?;

            let max_for_partition = partition_max_sizes.entry(partition_id).or_insert(0);
            if required > *max_for_partition {
                *max_for_partition = required;
            }

            partition_tensors
                .entry(partition_id)
                .or_default()
                .push(name.clone());

            tensors.insert(
                name.clone(),
                Arc::new(TensorDescriptor {
                    offset,
                    size,
                    shape: shape.into(),
                    stride: stride.into(),
                    dtype: dtype.into(),
                    partition_id,
                }),
            );
        }

        let mut partition_ids: Vec<PartitionId> = partition_max_sizes.keys().copied().collect();
        partition_ids.sort_unstable();
        let partition_ids: Arc<[PartitionId]> = partition_ids.into();

        let partitions: HashMap<PartitionId, PartitionPlan> = partition_ids
            .iter()
            .map(|&id| {
                let max_size = partition_max_sizes.get(&id).copied().unwrap_or(0);
                let tensor_names: Vec<Arc<str>> =
                    partition_tensors.get(&id).cloned().unwrap_or_default();
                (
                    id,
                    PartitionPlan {
                        partition_id: id,
                        max_required_size: max_size,
                        tensor_names: tensor_names.into(),
                    },
                )
            })
            .collect();

        let mut tensor_names_sorted: Vec<Arc<str>> = tensors.keys().cloned().collect();
        tensor_names_sorted.sort_unstable();
        let tensor_names_sorted: Arc<[Arc<str>]> = tensor_names_sorted.into();

        Ok(Index {
            tensors,
            partition_ids,
            partitions,
            tensor_names_sorted,
        })
    }

    /// Load index from file asynchronously.
    pub async fn load(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let engine = Tokio::new();
        let data = engine.read_file(path.as_ref()).await?;
        Self::from_bytes(&data)
    }

    /// Load index from file synchronously.
    pub fn load_sync(path: impl AsRef<Path>) -> ReaderResult<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(&data)
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

// Helper parsers for JSON values
fn parse_u64(value: &serde_json::Value) -> ReaderResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| ReaderError::ServerlessLlm("expected u64".into()))
}

fn parse_usize(value: &serde_json::Value) -> ReaderResult<usize> {
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| ReaderError::ServerlessLlm("expected usize".into()))
}

fn parse_string(value: &serde_json::Value) -> ReaderResult<String> {
    value
        .as_str()
        .map(String::from)
        .ok_or_else(|| ReaderError::ServerlessLlm("expected string".into()))
}

fn parse_usize_vec(value: &serde_json::Value) -> ReaderResult<Vec<usize>> {
    let arr = value
        .as_array()
        .ok_or_else(|| ReaderError::ServerlessLlm("expected array".into()))?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(
            v.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| ReaderError::ServerlessLlm("expected integer".into()))?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_index_with_6_fields() {
        let data = br#"{"tensor_b": [4, 2, [1,2], [2,1], "i8", 3]}"#;
        let index = Index::from_bytes(data).expect("parse index");
        let tensor_b = index.get("tensor_b").expect("tensor_b");
        assert_eq!(tensor_b.partition_id, PartitionId::new(3));
        assert_eq!(tensor_b.size, 2);
        assert_eq!(&*tensor_b.shape, &[1, 2]);
    }

    #[test]
    fn partition_ids_cached() {
        let data = br#"{
            "a": [0, 4, [2, 2], [2, 1], "f32", 2],
            "b": [4, 8, [2, 4], [4, 1], "f32", 0],
            "c": [12, 8, [2, 4], [4, 1], "f32", 2]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        assert_eq!(index.partition_ids(), &[PartitionId::new(0), PartitionId::new(2)]);
    }

    #[test]
    fn partition_max_size_computed() {
        let data = br#"{
            "a": [0, 4, [2, 2], [2, 1], "f32", 0],
            "b": [10, 20, [2, 4], [4, 1], "f32", 0]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        let p0 = index.partition(PartitionId::new(0)).expect("partition 0");
        assert_eq!(p0.max_required_size, 30);
    }

    #[test]
    fn empty_index_accessors() {
        let index = Index::new();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.partition_ids().is_empty());
        assert!(index.tensor_names().is_empty());
        assert!(index.get("anything").is_none());
        assert!(!index.contains("anything"));
        assert_eq!(index.iter().count(), 0);
    }

    #[test]
    fn default_is_empty() {
        let index = Index::default();
        assert!(index.is_empty());
    }

    #[test]
    fn from_bytes_malformed_json() {
        let result = Index::from_bytes(b"not json");
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_wrong_array_length() {
        let data = br#"{"tensor": [0, 4, [2], [1], "f32"]}"#;
        let result = Index::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn from_bytes_non_numeric_offset() {
        let data = br#"{"tensor": ["abc", 4, [2], [1], "f32", 0]}"#;
        let result = Index::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn tensor_names_sorted() {
        let data = br#"{
            "z_tensor": [0, 4, [2, 2], [2, 1], "f32", 0],
            "a_tensor": [4, 4, [2, 2], [2, 1], "f32", 0],
            "m_tensor": [8, 4, [2, 2], [2, 1], "f32", 1]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        let names: Vec<&str> = index.tensor_names().iter().map(|n| &**n).collect();
        assert_eq!(names, vec!["a_tensor", "m_tensor", "z_tensor"]);
    }

    #[test]
    fn contains_and_get() {
        let data = br#"{"weight": [0, 8, [2, 4], [4, 1], "f32", 0]}"#;
        let index = Index::from_bytes(data).expect("parse");
        assert!(index.contains("weight"));
        assert!(!index.contains("bias"));
        assert!(index.get("weight").is_some());
        assert!(index.get("bias").is_none());
    }

    #[test]
    fn iter_yields_all_tensors() {
        let data = br#"{
            "a": [0, 4, [2], [1], "f32", 0],
            "b": [4, 4, [2], [1], "f32", 0]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        assert_eq!(index.iter().count(), index.len());
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn tensors_in_same_partition() {
        let data = br#"{
            "a": [0, 4, [2], [1], "f32", 0],
            "b": [4, 8, [4], [1], "f32", 0]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        assert_eq!(index.partition_ids(), &[PartitionId::new(0)]);
        let p = index.partition(PartitionId::new(0)).unwrap();
        assert_eq!(p.tensor_names.len(), 2);
    }

    mod prop {
        use super::*;
        use proptest::prelude::*;

        fn arb_tensor_entry(name: &str) -> String {
            format!(r#""{name}": [0, 4, [2, 2], [2, 1], "f32", 0]"#)
        }

        proptest! {
            #[test]
            fn parse_preserves_tensor_count(count in 1usize..20) {
                let entries: Vec<String> = (0..count)
                    .map(|i| arb_tensor_entry(&format!("t_{i}")))
                    .collect();
                let json = format!("{{{}}}", entries.join(","));
                let index = Index::from_bytes(json.as_bytes()).unwrap();
                prop_assert_eq!(index.len(), count);
                prop_assert_eq!(index.tensor_names().len(), count);
            }

            #[test]
            fn names_always_sorted(count in 1usize..20) {
                let entries: Vec<String> = (0..count)
                    .map(|i| arb_tensor_entry(&format!("tensor_{:04}", count - i)))
                    .collect();
                let json = format!("{{{}}}", entries.join(","));
                let index = Index::from_bytes(json.as_bytes()).unwrap();
                let names: Vec<&str> = index.tensor_names().iter().map(|n| &**n).collect();
                for w in names.windows(2) {
                    prop_assert!(w[0] <= w[1], "names not sorted: {:?}", names);
                }
            }
        }
    }
}
