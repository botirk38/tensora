//! Hugging Face model id → local checkpoint paths (Hub cache + optional ServerlessLLM conversion).
//!
//! Nothing is written under a repository `fixtures/` directory; SafeTensors shards live in the
//! standard Hugging Face cache, and converted ServerlessLLM layouts sit under the OS cache.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::Repo;

use crate::recommended_partition_count;
use crate::WriterError;

/// Slug used for cache subdirectories (`org-name` lowercased).
pub fn filesystem_slug(model_id: &str) -> String {
    model_id.replace('/', "-").to_lowercase()
}

fn cache_root() -> PathBuf {
    std::env::var("TENSORA_CACHE_DIR")
        .map(PathBuf::from)
        .ok()
        .unwrap_or_else(|| {
            dirs::cache_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("tensora")
        })
}

/// Errors while resolving a Hub model to local paths.
#[derive(Debug, thiserror::Error)]
pub enum HfModelError {
    /// Hugging Face Hub API / download.
    #[error("Hugging Face Hub: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    /// Local I/O.
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),

    /// Conversion or validation.
    #[error("conversion: {0}")]
    Convert(#[from] WriterError),

    /// User-facing message.
    #[error("{0}")]
    Msg(String),
}

/// Directory containing downloaded `.safetensors` shards for `model_id` (Hugging Face cache layout).
///
/// Downloads missing shards via the Hub API. All shards share one snapshot directory.
pub fn ensure_safetensors_hub_dir(model_id: &str) -> Result<PathBuf, HfModelError> {
    let api = ApiBuilder::from_env().with_progress(true).build()?;
    let repo = Repo::model(model_id.to_string());
    let info = api.repo(repo.clone()).info()?;

    let shards: Vec<_> = info
        .siblings
        .into_iter()
        .filter(|s| s.rfilename.ends_with(".safetensors"))
        .collect();

    if shards.is_empty() {
        return Err(HfModelError::Msg(format!(
            "no .safetensors files listed for {model_id} on the Hub"
        )));
    }

    let mut parent: Option<PathBuf> = None;
    for s in &shards {
        let path = api.repo(repo.clone()).get(&s.rfilename)?;
        let p = path
            .parent()
            .ok_or_else(|| HfModelError::Msg("shard path has no parent".to_string()))?
            .to_path_buf();
        if let Some(ref prev) = parent {
            if prev != &p {
                return Err(HfModelError::Msg(
                    "safetensors shards resolved to different snapshot directories (unexpected)"
                        .to_string(),
                ));
            }
        } else {
            parent = Some(p);
        }
    }

    parent.ok_or_else(|| HfModelError::Msg("no safetensors directory".to_string()))
}

fn dir_safetensors_total_bytes(dir: &Path) -> Result<u64, HfModelError> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "safetensors") && path.is_file() {
            total += path.metadata()?.len();
        }
    }
    Ok(total)
}

/// ServerlessLLM artifact directory for `model_id`, under the OS cache (`tensora/<slug>/serverlessllm`).
///
/// Builds from `safetensors_dir` when `tensor_index.json` is absent (same partition heuristic as `convert`).
pub fn ensure_serverlessllm_cache_dir(
    model_id: &str,
    safetensors_dir: &Path,
) -> Result<PathBuf, HfModelError> {
    let slug = filesystem_slug(model_id);
    let out = cache_root().join(slug).join("serverlessllm");
    if out.join("tensor_index.json").is_file() {
        return Ok(out);
    }

    std::fs::create_dir_all(&out)?;

    let total = dir_safetensors_total_bytes(safetensors_dir)?;
    if total == 0 {
        return Err(HfModelError::Msg(format!(
            "no .safetensors files under {}",
            safetensors_dir.display()
        )));
    }
    let partitions = recommended_partition_count(total);
    if partitions == 0 {
        return Err(HfModelError::Msg(
            "partition count resolved to 0".to_string(),
        ));
    }

    crate::convert_safetensors_to_serverlessllm_sync(
        safetensors_dir
            .to_str()
            .ok_or_else(|| HfModelError::Msg("safetensors path is not UTF-8".to_string()))?,
        out.to_str()
            .ok_or_else(|| HfModelError::Msg("output path is not UTF-8".to_string()))?,
        partitions,
    )?;

    Ok(out)
}
