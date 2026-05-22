# Benchmarks

Criterion.rs performance benchmarks for all Tensora I/O paths.

## Suites

| File | Measures |
|------|----------|
| `safetensors.rs` | Full-model load, tensor access patterns, native baseline |
| `serverlessllm.rs` | Full-model load, tensor access, mmap page-touch |
| `conversion.rs` | SafeTensors → ServerlessLLM pipeline |
| `backends.rs` | Raw I/O micro-benchmarks (single-shard, batch, range) |
| `bench_util.rs` | Shared helpers (model resolution, throughput) |

All benchmarks report throughput (bytes/sec) alongside wall-clock time.

## Running

```bash
export TENSORA_MODEL_ID=openai-community/gpt2

cargo bench                        # All suites
cargo bench --bench safetensors    # Single suite
cargo bench -- sync                # Filter by backend name
```

## Configuration

- 10 samples, 15s measurement window, 3s warmup
- Forces data materialization (mmap benchmarks touch all pages)
- Models resolved via HuggingFace Hub cache
