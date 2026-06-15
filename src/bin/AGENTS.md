# AGENTS.md — CLI Binaries

## Binaries

- `profile` — performance measurement (cold/warm cache, all storage engines)
- `demo` — interactive demonstrations
- `convert` — format conversion tool

## Conventions

- All binaries use `clap` for argument parsing
- Model resolution via `--model-id` (HuggingFace Hub)
- Use `--release` for meaningful timing numbers
- Config structs in dedicated `config.rs` files

## Do NOT

- Add new binaries without updating `Cargo.toml` `[[bin]]` entries
- Use `println!` for structured output (use proper formatting)
