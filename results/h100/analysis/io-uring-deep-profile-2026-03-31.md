## H100 io_uring bottleneck profile

Commands were run on `root@86.38.238.114` after manual cache drop:

```bash
sync
echo 3 > /proc/sys/vm/drop_caches
```

### SafeTensors `qwen-qwen3-0.6b` cold

`perf stat`

- `sync`: 455.96 ms, `task-clock` 2391.34 ms, `3.886 CPUs utilized`, `975` context switches
- `async`: 1259.45 ms, `task-clock` 1247.59 ms, `0.847 CPUs utilized`, `306` context switches
- `io-uring`: 1015.26 ms, `task-clock` 1138.74 ms, `0.983 CPUs utilized`, `27` context switches
- `default`: 436.95 ms, `task-clock` 2213.16 ms, `3.584 CPUs utilized`, `1159` context switches

`strace -c`

- `sync`: `91.38%` of syscall time in `read`, `27` reads total, many helper threads attached
- `io-uring`: `91.36%` of syscall time in `io_uring_enter`, only `8` enters, almost no thread-level parallelism

### SafeTensors `qwen-qwen3-8b` cold

`perf stat`

- `sync`: 4032.09 ms, `40030.99 ms task-clock`, `7.824 CPUs utilized`, `17663` context switches
- `io-uring`: 10275.41 ms, `9907.74 ms task-clock`, `0.877 CPUs utilized`, `180` context switches

### ServerlessLLM `qwen-qwen3-8b` cold

`perf stat`

- `sync`: 7535.97 ms, `32774.10 ms task-clock`, `3.817 CPUs utilized`, `2167` context switches
- `io-uring`: 28894.67 ms, `24695.51 ms task-clock`, `0.825 CPUs utilized`, `369` context switches

## Diagnosis

### 1. `io_uring` is still effectively single-core

- `SafeTensors` 0.6B: `0.983 CPUs utilized` for `io-uring` vs `3.886` for `sync`
- `SafeTensors` 8B: `0.877 CPUs utilized` for `io-uring` vs `7.824` for `sync`
- `ServerlessLLM` 8B: `0.825 CPUs utilized` for `io-uring` vs `3.817` for `sync`

This means the backend is not exploiting concurrency the way the sync backend does.

Relevant code:

- `src/backends/io_uring.rs:131` submits until SQ is full, but then immediately `submit_and_wait(1)` at `src/backends/io_uring.rs:155`
- `src/backends/io_uring.rs:218` and `src/backends/io_uring.rs:286` do the same for whole-file buffered/direct loads
- `src/formats/safetensors/model.rs:433` still loads shards one by one through a shared reader; there is no format-level shard batch path

### 2. syscall overhead still dominates `io_uring`

From `strace -c` on 0.6B:

- `sync`: `91.38%` in plain `read`
- `io-uring`: `91.36%` in `io_uring_enter`

So the backend still spends most syscall time entering the kernel instead of sustaining a deep queue.

### 3. `load_range_batch` is not really exploiting grouped files yet

`src/backends/io_uring.rs:110` groups requests by file only to open files once.
After that, requests are still driven by original request order at `src/backends/io_uring.rs:133`, not by a grouped or coalesced submission plan.

That means:

- file grouping is not being used to shape submission order
- adjacent requests are not coalesced
- the ring is fed request-by-request rather than with a more optimized plan

### 4. direct-I/O path is suspiciously wrong

`src/backends/io_uring.rs:270` uses `open_direct_read_sync(path)` but writes into `OwnedBytes::from_vec(vec![0u8; file_size])` at `src/backends/io_uring.rs:272`.

That buffer is not guaranteed to be block-aligned, unlike the aligned buffers used by `sync` in `src/backends/sync_io.rs:49` and `src/backends/sync_io.rs:152`.

So the current `io_uring` direct path is at best suboptimal and at worst semantically wrong for true O_DIRECT expectations.

### 5. SafeTensors format path still favors sync parallelism massively

`sync` uses parallel chunking and helper threads in `src/backends/sync_io.rs:65` onward, which matches the observed multi-core use.

`io_uring` uses one ring and one submission loop, but the current implementation is not enough to match that parallel throughput.

### 6. ServerlessLLM shows the same problem, amplified by fragmentation

`src/formats/serverlessllm/model.rs:292` does use `load_range_batch`, so the batch path is being exercised.
But the results are still much worse than sync, which points back to the batch engine itself rather than caller misuse.

## Main bottlenecks to fix next

1. Stop using `submit_and_wait(1)` as the steady-state batch strategy
2. Build a real submission schedule from grouped requests instead of only grouping for file opens
3. Add request coalescing for adjacent ranges in the same file
4. Fix direct-I/O buffers to use aligned storage, not `Vec<u8>`
5. Add a true batched shard path for SafeTensors if `io_uring` is to compete on sharded models

## Recommendation

Until those are fixed, `default` should not choose `io-uring` on H100. The measured evidence says it loses badly across both formats and multiple sizes, and the current implementation is bottlenecked by low parallelism plus `io_uring_enter` overhead.
