# convert

Converts SafeTensors checkpoints to ServerlessLLM partitioned format.

## Usage

```bash
cargo run --bin convert -- <INPUT_DIR> <OUTPUT_DIR> [OPTIONS]
```

### Options

- `-p, --partitions <N>` — Number of partitions (default: auto based on model size)
- `-e, --engine <ENGINE>` — I/O backend: `default`, `sync`, `tokio`, `io-uring`

## Examples

```bash
cargo run --bin convert -- ./model_dir ./output
cargo run --bin convert -- ./model_dir ./output --partitions 8 --engine sync
```

## Output

```
output/
├── metadata.json       # Tensor index with partition assignments
├── tensor.data_0       # Partition 0
├── tensor.data_1       # Partition 1
└── tensor.data_{N-1}   # Last partition
```

Default partition count: `max(1, ceil(model_bytes / 512 MiB))`.
