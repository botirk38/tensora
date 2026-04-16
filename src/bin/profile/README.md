# profile

Profiling harness for measuring tensora loader performance. Pass a **Hugging Face model id**; SafeTensors shards are resolved through the Hub client cache, and ServerlessLLM layouts are built under the OS cache (`tensora/<slug>/serverlessllm`).

## Usage

```bash
cargo run --release --bin profile -- <COMMAND> <CASE> --model-id <ORG/NAME> [OPTIONS]
```

## Commands

### SafeTensors

```bash
cargo run --release --bin profile -- safetensors <CASE> --model-id <HF/MODEL> [--iterations <N>]
```

### ServerlessLLM

```bash
cargo run --release --bin profile -- serverlessllm <CASE> --model-id <HF/MODEL> [--iterations <N>]
```

**Cases:** `default`, `sync`, `async`, `mmap`, `io-uring` (Linux only).

## Options

- `--model-id` — **Required.** Hugging Face repository id (e.g. `Qwen/Qwen3-8B`).
- `-i, --iterations` — Repeat count (default `1`).

## Examples

```bash
cargo build --release --bin profile
cargo run --release --bin profile -- safetensors default --model-id openai-community/gpt2
cargo run --release --bin profile -- serverlessllm sync --model-id Qwen/Qwen3-0.6B --iterations 3
```

Cold-cache measurements: run `sync` and `echo 3 > /proc/sys/vm/drop_caches` before `profile` as in the paper’s Experimental Setup.
