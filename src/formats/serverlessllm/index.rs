//! ServerlessLLM index parsing and metadata.
//!
//! Redesigned for compiled metadata - all partition info computed once at parse time.

use crate::formats::error::{LoadError, LoadResult};
use crate::formats::tensor::Dtype;
use std::collections::HashMap;
use std::sync::Arc;
use std::str::FromStr;

use super::ids::PartitionId;

/// Iterator type for tensor names.
pub type TensorNamesIter<'a> = std::iter::Map<std::slice::Iter<'a, Arc<str>>, fn(&'a Arc<str>) -> &'a str>;

// ============================================================================
// TensorDescriptor
// ============================================================================

/// Compact tensor descriptor for load planning.
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    offset: u64,
    size: usize,
    shape: Arc<[usize]>,
    stride: Arc<[usize]>,
    dtype: Dtype,
    partition_id: PartitionId,
}

impl TensorDescriptor {
    /// Returns the offset within the partition.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the size in bytes.
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the shape.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the stride.
    #[inline]
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    /// Returns the dtype.
    #[inline]
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Returns the partition ID.
    #[inline]
    #[must_use]
    pub fn partition_id(&self) -> PartitionId {
        self.partition_id
    }
}

// ============================================================================
// PartitionPlan
// ============================================================================

/// Precomputed partition metadata for load planning.
#[derive(Debug, Clone)]
pub struct PartitionPlan {
    partition_id: PartitionId,
    max_required_size: u64,
    tensor_names: Arc<[Arc<str>]>,
}

impl PartitionPlan {
    /// Returns the partition ID.
    #[inline]
    #[must_use]
    pub fn partition_id(&self) -> PartitionId {
        self.partition_id
    }

    /// Returns the maximum required size for this partition.
    #[inline]
    #[must_use]
    pub fn max_required_size(&self) -> u64 {
        self.max_required_size
    }

    /// Returns an iterator over tensor names in this partition.
    pub fn tensor_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.tensor_names.iter().map(|n| n.as_ref())
    }
}

// ============================================================================
// Index
// ============================================================================

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
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name).map(|arc| arc.as_ref())
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

    /// Returns an iterator over tensor names.
    pub fn tensor_names(&self) -> TensorNamesIter<'_> {
        fn arc_to_str(arc: &Arc<str>) -> &str {
            arc.as_ref()
        }
        self.tensor_names_sorted.iter().map(arc_to_str)
    }

    /// Internal iterator for model trait integration.
    #[allow(dead_code)]
    pub(crate) fn tensor_names_iter(&self) -> TensorNamesIter<'_> {
        fn arc_to_str(arc: &Arc<str>) -> &str {
            arc.as_ref()
        }
        self.tensor_names_sorted.iter().map(arc_to_str)
    }

    /// Returns an iterator over tensor names and descriptors.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorDescriptor)> {
        self.tensors.iter().map(|(k, v)| (k.as_ref(), v.as_ref()))
    }

    /// Returns sorted partition IDs (cached, computed once at parse time).
    #[inline]
    #[must_use]
    pub fn partition_ids(&self) -> &[PartitionId] {
        &self.partition_ids
    }

    /// Returns the partition plan for a given partition ID.
    #[inline]
    #[must_use]
    pub fn partition(&self, id: PartitionId) -> Option<&PartitionPlan> {
        self.partitions.get(&id)
    }

    /// Parse index from raw bytes.
    pub fn from_bytes(data: &[u8]) -> LoadResult<Self> {
        let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(data)
            .map_err(|err| LoadError::ServerlessLlm(format!("JSON parse error: {err}")))?;

        let mut tensors = HashMap::with_capacity(raw.len());
        let mut partition_max_sizes: HashMap<PartitionId, u64> = HashMap::new();
        let mut partition_tensors: HashMap<PartitionId, Vec<Arc<str>>> = HashMap::new();

        for (name, value) in raw {
            let name: Arc<str> = name.as_str().into();
            let entry = TensorIndexEntryJson::new(&value)?;

            let offset = entry.offset()?;
            let size = entry.size()?;
            let size_u64 = size as u64;
            let shape = entry.shape()?;
            let stride = entry.stride()?;
            let dtype_str = entry.dtype_str()?;
            let dtype = Dtype::from_str(&dtype_str).map_err(|err| {
                LoadError::InvalidMetadata(format!("invalid dtype '{}': {err}", dtype_str))
            })?;
            let partition_id = entry.partition_id()?;
            let required = offset
                .checked_add(size_u64)
                .ok_or_else(|| LoadError::OffsetOverflow {
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
                    dtype,
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
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// JSON Parser Entity
// ============================================================================

/// JSON parser for a single tensor index entry.
///
/// Parses the 6-element array format: [offset, size, shape, stride, dtype, partition_id].
struct TensorIndexEntryJson<'a> {
    fields: &'a [serde_json::Value],
}

impl<'a> TensorIndexEntryJson<'a> {
    /// Create a new entry parser from a JSON value.
    fn new(value: &'a serde_json::Value) -> LoadResult<Self> {
        let arr = value.as_array().ok_or_else(|| {
            LoadError::ServerlessLlm("tensor entry must be an array".to_string())
        })?;

        let arr = arr.as_slice();
        if arr.len() != 6 {
            return Err(LoadError::ServerlessLlm(format!(
                "tensor entry must have exactly 6 elements, got {}",
                arr.len()
            )));
        }

        Ok(Self { fields: arr })
    }

    /// Parse offset (element 0).
    fn offset(&self) -> LoadResult<u64> {
        self.fields[0]
            .as_u64()
            .ok_or_else(|| LoadError::ServerlessLlm("expected u64 for offset".into()))
    }

    /// Parse size (element 1).
    fn size(&self) -> LoadResult<usize> {
        self.fields[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| LoadError::ServerlessLlm("expected usize for size".into()))
    }

    /// Parse shape (element 2).
    fn shape(&self) -> LoadResult<Vec<usize>> {
        parse_usize_vec(&self.fields[2], "shape")
    }

    /// Parse stride (element 3).
    fn stride(&self) -> LoadResult<Vec<usize>> {
        parse_usize_vec(&self.fields[3], "stride")
    }

    /// Parse dtype string (element 4).
    fn dtype_str(&self) -> LoadResult<String> {
        self.fields[4]
            .as_str()
            .map(String::from)
            .ok_or_else(|| LoadError::ServerlessLlm("expected string for dtype".into()))
    }

    /// Parse partition_id (element 5).
    fn partition_id(&self) -> LoadResult<PartitionId> {
        let id = self.fields[5]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| LoadError::ServerlessLlm("expected usize for partition_id".into()))?;
        Ok(PartitionId::new(id))
    }
}

/// Parse a vector of unsigned integers from a JSON value.
fn parse_usize_vec(value: &serde_json::Value, field_name: &str) -> LoadResult<Vec<usize>> {
    let arr = value.as_array().ok_or_else(|| {
        LoadError::ServerlessLlm(format!("expected array for {}", field_name))
    })?;

    arr.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    LoadError::ServerlessLlm(format!(
                        "expected integer for {} element",
                        field_name
                    ))
                })
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_index_with_6_fields() {
        let data = br#"{"tensor_b": [4, 2, [1,2], [2,1], "i8", 3]}"#;
        let index = Index::from_bytes(data).expect("parse index");
        let tensor_b = index.get("tensor_b").expect("tensor_b");
        assert_eq!(tensor_b.partition_id(), PartitionId::new(3));
        assert_eq!(tensor_b.size(), 2);
        assert_eq!(tensor_b.shape(), &[1, 2]);
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
            "b": [4, 8, [2, 4], [4, 1], "f32", 0],
            "c": [12, 8, [2, 4], [4, 1], "f32", 0]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        let plan = index.partition(PartitionId::new(0)).expect("partition 0");
        assert_eq!(plan.max_required_size(), 20);
    }

    #[test]
    fn tensor_names_sorted() {
        let data = br#"{"c": [0, 4, [1], [1], "f32", 0], "a": [0, 4, [1], [1], "f32", 0], "b": [0, 4, [1], [1], "f32", 0]}"#;
        let index = Index::from_bytes(data).expect("parse");
        let names: Vec<&str> = index.tensor_names().collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[test]
    fn descriptor_accessors_work() {
        let data = br#"{"test": [8, 16, [2, 2], [2, 1], "f64", 1]}"#;
        let index = Index::from_bytes(data).expect("parse");
        let desc = index.get("test").expect("test tensor");
        assert_eq!(desc.offset(), 8);
        assert_eq!(desc.size(), 16);
        assert_eq!(desc.shape(), &[2, 2]);
        assert_eq!(desc.stride(), &[2, 1]);
        assert_eq!(desc.dtype(), Dtype::F64);
        assert_eq!(desc.partition_id(), PartitionId::new(1));
    }

    #[test]
    fn get_returns_none_for_missing() {
        let data = br#"{"exists": [0, 4, [1], [1], "f32", 0]}"#;
        let index = Index::from_bytes(data).expect("parse");
        assert!(index.get("missing").is_none());
        assert!(!index.contains("missing"));
        assert!(index.contains("exists"));
    }

    #[test]
    fn iter_provides_name_descriptor_pairs() {
        let data = br#"{"a": [0, 4, [1], [1], "f32", 0], "b": [0, 8, [2], [1], "f64", 1]}"#;
        let index = Index::from_bytes(data).expect("parse");
        let mut pairs: Vec<(&str, Dtype)> = index
            .iter()
            .map(|(name, desc)| (name, desc.dtype()))
            .collect();
        pairs.sort_by_key(|(name, _)| *name);
        assert_eq!(pairs, vec![("a", Dtype::F32), ("b", Dtype::F64)]);
    }

    #[test]
    fn empty_index() {
        let index = Index::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        assert!(index.tensor_names().next().is_none());
    }

    #[test]
    fn tensor_entry_json_parses_all_fields() {
        let json = serde_json::json!([16, 32, [4, 8], [8, 1], "f16", 2]);
        let entry = TensorIndexEntryJson::new(&json).expect("parse");
        assert_eq!(entry.offset().unwrap(), 16);
        assert_eq!(entry.size().unwrap(), 32);
        assert_eq!(entry.shape().unwrap(), vec![4, 8]);
        assert_eq!(entry.stride().unwrap(), vec![8, 1]);
        assert_eq!(entry.dtype_str().unwrap(), "f16");
        assert_eq!(entry.partition_id().unwrap(), PartitionId::new(2));
    }

    #[test]
    fn tensor_entry_json_rejects_wrong_element_count() {
        let json = serde_json::json!([1, 2, 3]); // only 3 elements
        assert!(TensorIndexEntryJson::new(&json).is_err());
    }

    #[test]
    fn tensor_entry_json_rejects_non_array() {
        let json = serde_json::json!("not an array");
        assert!(TensorIndexEntryJson::new(&json).is_err());
    }
}
