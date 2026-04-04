# Final Year Project Submission

## TensorStore: Adaptive Checkpoint Loading for LLM Inference

**Author:** Botir Khaltaev  
**Supervisor:** Dr. Zafeirios Zafeirakopoulos  
**Date:** April 2026

## Project Overview

This project implements a Rust-based tensor storage library with Python bindings that provides adaptive, backend-aware checkpoint loading for Large Language Models (LLMs). Unlike traditional approaches that rely on a fixed I/O backend, TensorStore automatically selects between synchronous POSIX, asynchronous io_uring, and memory-mapped access patterns based on workload characteristics.

## Files

- `FYP Final Report | Botir Khaltaev.pdf` - Final project report

## Key Features

- **Adaptive Backend Selection**: Automatically chooses optimal I/O backend (sync, io_uring, async) based on runtime metrics
- **Multi-Format Support**: SafeTensors and ServerlessLLM checkpoint formats
- **Framework Integrations**: PyTorch and vLLM inference pipelines
- **Python Bindings**: Clean Python API via PyO3

## Demo

Live demonstration available at: https://huggingface.co/spaces/Botir/tensor-store-demo

## Running the Examples

```bash
# PyTorch example
cd bindings/python
uv sync --group dev --group torch
uv run python examples/pytorch.py gpt2 --prompt "Hello, world!"

# vLLM example
uv sync --group dev --group vllm
uv run python examples/vllm_infer.py --model gpt2 --prompt "Hello, world!"
```

## Repository Structure

```
PROJECT/
├── src/                    # Rust source code
├── bindings/python/        # Python bindings
│   ├── tensor_store_py/    # Python package
│   ├── examples/           # Runnable examples
│   └── benchmarks/         # Benchmark suite
├── report/                 # Project report (LaTeX)
└── scripts/                # Utility scripts
```