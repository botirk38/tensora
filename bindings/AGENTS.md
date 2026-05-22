# AGENTS.md — Bindings

## Purpose

Language bindings exposing the Tensora Rust library to other runtimes.

## Current Bindings

- `python/` — `tensora_py` package (Python ≥ 3.12, maturin + uv)

## Conventions

- Each binding gets its own subdirectory with independent build tooling
- Native extensions link against the workspace root crate
- Framework-specific code (PyTorch, TensorFlow) uses lazy imports
- Benchmarks and tests are self-contained within each binding directory

## Adding a New Binding

1. Create `bindings/<language>/` with its build system
2. Add a README.md and AGENTS.md
3. Reference the root crate as a dependency
4. Add integration tests that exercise the full load path
