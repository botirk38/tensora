//! ServerlessLLM index parsing and metadata.
//!
//! All partition info is computed once at parse time and cached.

use crate::formats::error::{LoadError, LoadResult};
use crate::formats::tensor::Dtype;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;

use super::ids::PartitionId;
use super::tensor::TensorEntry;

/// Iterator type for tensor names.
pub type TensorNamesIter<'a> =
    std::iter::Map<std::slice::Iter<'a, Arc<str>>, fn(&'a Arc<str>) -> &'a str>;

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

    /// Returns the minimum required byte capacity for this partition.
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
    tensors: HashMap<Arc<str>, Arc<TensorEntry>>,
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

    /// Gets the [`PartitionedTensor`] metadata for a tensor by name.
    #[inline]
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name).map(|arc| arc.as_ref())
    }

    /// Returns `true` if a tensor with the given name exists.
    #[inline]
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Returns the number of tensors in the index.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns `true` when the index has no tensors.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Returns an iterator over tensor names in sorted order.
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

    /// Returns an iterator over `(name, PartitionedTensor)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorEntry)> {
        self.tensors.iter().map(|(k, v)| (k.as_ref(), v.as_ref()))
    }

    /// Returns sorted partition IDs (cached at parse time).
    #[inline]
    #[must_use]
    pub fn partition_ids(&self) -> &[PartitionId] {
        &self.partition_ids
    }

    /// Returns the [`PartitionPlan`] for a given partition ID.
    #[inline]
    #[must_use]
    pub fn partition(&self, id: PartitionId) -> Option<&PartitionPlan> {
        self.partitions.get(&id)
    }

    /// Parse a ServerlessLLM `tensor_index.json` from raw bytes.
    ///
    /// Each tensor entry is a 6-element JSON array:
    /// `[offset, size, [shape...], [stride...], "dtype", partition_id]`.
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
            let size_usize = entry.size()?;
            let size = size_usize as u64;
            let shape = entry.shape()?;
            let stride = entry.stride()?;
            let dtype_str = entry.dtype_str()?;
            let dtype = Dtype::from_str(&dtype_str).map_err(|err| {
                LoadError::InvalidMetadata(format!("invalid dtype '{}': {err}", dtype_str))
            })?;
            let partition_id = entry.partition_id()?;

            let required = offset
                .checked_add(size)
                .ok_or_else(|| LoadError::OffsetOverflow {
                    name: name.to_string(),
                })?;

            let max = partition_max_sizes.entry(partition_id).or_insert(0);
            if required > *max {
                *max = required;
            }

            partition_tensors
                .entry(partition_id)
                .or_default()
                .push(name.clone());

            let pt = TensorEntry::from_parts(offset, size, shape, stride, dtype, partition_id)
                .map_err(|e| LoadError::InvalidMetadata(e.to_string()))?;

            tensors.insert(name.clone(), Arc::new(pt));
        }

        let mut partition_ids: Vec<PartitionId> = partition_max_sizes.keys().copied().collect();
        partition_ids.sort_unstable();
        let partition_ids: Arc<[PartitionId]> = partition_ids.into();

        let partitions: HashMap<PartitionId, PartitionPlan> = partition_ids
            .iter()
            .map(|&id| {
                let max_size = partition_max_sizes.get(&id).copied().unwrap_or(0);
                let names: Vec<Arc<str>> = partition_tensors.get(&id).cloned().unwrap_or_default();
                (
                    id,
                    PartitionPlan {
                        partition_id: id,
                        max_required_size: max_size,
                        tensor_names: names.into(),
                    },
                )
            })
            .collect();

        let mut tensor_names_sorted: Vec<Arc<str>> = tensors.keys().cloned().collect();
        tensor_names_sorted.sort_unstable();

        Ok(Index {
            tensors,
            partition_ids,
            partitions,
            tensor_names_sorted: tensor_names_sorted.into(),
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
/// Format: `[offset, size, [shape...], [stride...], "dtype", partition_id]`.
struct TensorIndexEntryJson<'a> {
    fields: &'a [serde_json::Value],
}

impl<'a> TensorIndexEntryJson<'a> {
    fn new(value: &'a serde_json::Value) -> LoadResult<Self> {
        let arr = value
            .as_array()
            .ok_or_else(|| LoadError::ServerlessLlm("tensor entry must be an array".to_string()))?;
        let arr = arr.as_slice();
        if arr.len() != 6 {
            return Err(LoadError::ServerlessLlm(format!(
                "tensor entry must have exactly 6 elements, got {}",
                arr.len()
            )));
        }
        Ok(Self { fields: arr })
    }

    fn offset(&self) -> LoadResult<u64> {
        self.fields[0]
            .as_u64()
            .ok_or_else(|| LoadError::ServerlessLlm("expected u64 for offset".into()))
    }

    fn size(&self) -> LoadResult<usize> {
        self.fields[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| LoadError::ServerlessLlm("expected usize for size".into()))
    }

    fn shape(&self) -> LoadResult<Vec<usize>> {
        parse_usize_vec(&self.fields[2], "shape")
    }

    fn stride(&self) -> LoadResult<Vec<usize>> {
        parse_usize_vec(&self.fields[3], "stride")
    }

    fn dtype_str(&self) -> LoadResult<String> {
        self.fields[4]
            .as_str()
            .map(String::from)
            .ok_or_else(|| LoadError::ServerlessLlm("expected string for dtype".into()))
    }

    fn partition_id(&self) -> LoadResult<PartitionId> {
        let id = self.fields[5]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| LoadError::ServerlessLlm("expected usize for partition_id".into()))?;
        Ok(PartitionId::new(id))
    }
}

fn parse_usize_vec(value: &serde_json::Value, field_name: &str) -> LoadResult<Vec<usize>> {
    let arr = value
        .as_array()
        .ok_or_else(|| LoadError::ServerlessLlm(format!("expected array for {field_name}")))?;
    arr.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    LoadError::ServerlessLlm(format!("expected integer for {field_name} element"))
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
        let t = index.get("tensor_b").expect("tensor_b");
        assert_eq!(t.partition_id(), PartitionId::new(3));
        assert_eq!(t.size(), 2);
        assert_eq!(t.shape(), &[1, 2]);
    }

    #[test]
    fn partition_ids_sorted() {
        let data = br#"{
            "a": [0, 4, [2, 2], [2, 1], "f32", 2],
            "b": [4, 8, [2, 4], [4, 1], "f32", 0],
            "c": [12, 8, [2, 4], [4, 1], "f32", 2]
        }"#;
        let index = Index::from_bytes(data).expect("parse");
        assert_eq!(
            index.partition_ids(),
            &[PartitionId::new(0), PartitionId::new(2)]
        );
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
    fn accessor_delegation() {
        let data = br#"{"test": [8, 16, [2, 2], [2, 1], "f64", 1]}"#;
        let index = Index::from_bytes(data).expect("parse");
        let t = index.get("test").expect("test tensor");
        assert_eq!(t.offset(), 8);
        assert_eq!(t.size(), 16);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.stride(), &[2, 1]);
        assert_eq!(t.dtype(), Dtype::F64);
        assert_eq!(t.partition_id(), PartitionId::new(1));
    }

    #[test]
    fn empty_index_parses() {
        let index = Index::from_bytes(b"{}").expect("empty");
        assert!(index.is_empty());
    }

    #[test]
    fn malformed_entry_errors() {
        let data = br#"{"bad": [0, 4, [1], [1], "f32"]}"#; // 5 elements
        assert!(Index::from_bytes(data).is_err());
    }
}
