# tensor_store_py Benchmarks

Benchmarks for the Python bindings. Uses pytest-benchmark with synthetic GPT-2-like fixtures.

## Prerequisites

- `uv sync` (with dev deps) so the package and pytest-benchmark are available

## Run benchmarks

```bash
uv run pytest benchmarks/bench_safetensors.py benchmarks/bench_serverlessllm.py --benchmark-only
```

## Structure

- **bench_safetensors.py** – SafeTensors: load_safetensors_sync, open+get_tensor (sync, mmap, async)
- **bench_serverlessllm.py** – ServerlessLLM: same operations

Each operation has **warm** and **cold** variants:

- **Warm**: Data in page cache (typical repeated access).
- **Cold**: `posix_fadvise(DONTNEED)` before each run to hint the kernel to drop pages (Unix-only).

All open+get_tensor benchmarks **touch every page** (`.sum().item()` on each tensor) so mmap results reflect real page faults, not just mapping.
