# AGENTS.md

Guidelines for AI agents working on the Tensora codebase.

## Project Overview

Tensora is an adaptive checkpoint loading framework for LLMs, written in Rust with Python bindings. It accompanies an academic paper.

## Build & Test

```bash
cargo build --release
cargo test --lib --locked
cargo clippy --lib --locked -- -D warnings
```

## Key Conventions

- **Rust edition:** 2024, minimum `rustc` 1.92
- **Error handling:** Use `thiserror` for library errors; `ReaderError`/`WriterError` in `formats::error`
- **Async runtime:** Tokio (multi-thread) for all async paths
- **Platform gating:** Use `#[cfg(target_os = "linux")]` for io_uring and O_DIRECT code
- **No `unsafe`** outside of platform-specific I/O backend internals
- **Tests:** Unit tests inline (`#[cfg(test)]`), integration tests under `tests/` if needed
- **Naming:** Modules match their directory structure; public re-exports in `lib.rs`

## Architecture Rules

1. **I/O backends** handle raw I/O only — no format awareness
2. **Formats** parse and serialize — no I/O strategy decisions
3. **Converters** orchestrate full pipelines — use formats + I/O backends together
4. **Heuristics** live in the format `model.rs` files (storage-engine selection logic)

## File Organization

- One module per file (no multi-thousand-line files)
- `mod.rs` files re-export public API only
- Platform-specific code in dedicated files with `cfg` gates

## Python Bindings

- Package: `tensora_py` (under `bindings/python/`)
- Build: `maturin develop --release`
- Python ≥ 3.12, managed with `uv`

## Do NOT

- Modify benchmark result files under `results/`
- Change the paper LaTeX without explicit instruction
- Add dependencies without justification
- Use `Any` or dynamic dispatch where static dispatch works
- Skip `cargo clippy` before committing
