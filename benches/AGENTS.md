# AGENTS.md — Benchmarks

## Purpose

Criterion.rs benchmarks measuring I/O backend performance on real model checkpoints.

## Running

```bash
export TENSORA_MODEL_ID=openai-community/gpt2
cargo bench
```

## Conventions

- 10 samples, 15s measurement, 3s warmup
- Report throughput (bytes/sec) alongside wall-clock
- Force full data materialization (no lazy mmap shortcuts)
- Use `bench_util.rs` helpers for model resolution

## Adding a Benchmark

1. Create `benches/<name>.rs`
2. Add `[[bench]]` entry in root `Cargo.toml` with `harness = false`
3. Use `criterion_group!` / `criterion_main!` macros
4. Report throughput via `Throughput::Bytes`

## Do NOT

- Use tiny synthetic data (always use real model shards)
- Skip warmup iterations
- Benchmark debug builds
