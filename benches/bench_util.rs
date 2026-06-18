//! Shared helpers for Criterion benchmarks.
//!
//! Provides model resolution, throughput computation, and helpers
//! for I/O-heavy benchmark workloads.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use tensora::formats::traits::{Model, Tensor};

pub const ENV_VAR: &str = "TENSORA_MODEL_ID";

pub fn resolve_safetensors_model() -> (String, PathBuf) {
    let id = std::env::var(ENV_VAR).unwrap_or_else(|_| {
        eprintln!("Set {ENV_VAR} to a Hugging Face model id (e.g. openai-community/gpt2).");
        std::process::exit(1);
    });
    let dir = tensora::hf_model::ensure_safetensors_hub_dir(&id).unwrap_or_else(|e| {
        eprintln!("Could not resolve model {id}: {e}");
        std::process::exit(1);
    });
    (id, dir)
}

pub fn resolve_serverlessllm_model() -> (String, PathBuf, PathBuf) {
    let (id, st_dir) = resolve_safetensors_model();
    let sllm_dir =
        tensora::hf_model::ensure_serverlessllm_cache_dir(&id, &st_dir).unwrap_or_else(|e| {
            eprintln!("Could not build ServerlessLLM layout for {id}: {e}");
            std::process::exit(1);
        });
    (id, st_dir, sllm_dir)
}

pub fn safetensors_total_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "safetensors") && path.is_file() {
                total += path.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

pub fn serverlessllm_total_bytes(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("tensor.data_") && path.is_file() {
                total += path.metadata().map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    total
}

pub fn first_safetensors_shard(dir: &Path) -> PathBuf {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read safetensors dir")
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension().is_some_and(|e| e == "safetensors") && p.is_file() {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    shards.sort();
    shards
        .into_iter()
        .next()
        .expect("at least one .safetensors shard")
}

/// Force materialization of all tensor data, returning total bytes touched.
pub fn touch_all_tensors<M: Model>(model: &M) -> usize {
    let mut total = 0usize;
    for name in model.tensor_names() {
        if let Some(t) = model.tensor(name.as_ref()) {
            let data = t.data();
            if !data.is_empty() {
                std::hint::black_box(data[0]);
                std::hint::black_box(data[data.len() - 1]);
            }
            total += data.len();
        }
    }
    total
}

pub fn model_slug(model_id: &str) -> String {
    model_id.replace('/', "-").to_lowercase()
}

pub fn collect_shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}
