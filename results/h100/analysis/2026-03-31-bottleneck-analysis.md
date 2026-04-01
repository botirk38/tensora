# H100 Profiling Analysis — 2026-03-31

## Environment
- **VM**: root@86.38.238.114
- **CPU**: AMD EPYC 9654 96-Core Processor
- **RAM**: 1.0 TiB
- **GPU**: NVIDIA H100 80GB HBM3
- **Kernel**: 6.8.0-100-generic
- **Rust**: 1.92.0
- **io-uring**: 0.7 (crates.io)

## io_uring Across Model Sizes

### Cold-Cache (1st iteration)
| Model | io_uring (ms) | sync (ms) | async (ms) | default (ms) | Winner |
|---|---|---|---|---|---|
| 0.6B (1.5GB) | 742 | 399 | 771 | 100 | default→sync |
| 3B (6.1GB) | 2986 | 1337 | 2490 | 278 | default→sync |
| 4B (8.0GB) | 3931 | 1801 | 2093 | 350 | default→sync |
| 8B (16.4GB) | 7885 | 3761 | 2628 | 2730 | async |
| 14B (29.5GB) | 21528 | 5877 | 4652 | 4950 | async |
| 32B (65.5GB) | 47736 | 11085 | 25515 | 27723 | sync |

### Warm-Cache (min of 3 iters)
| Model | io_uring (ms) | sync (ms) | async (ms) | default (ms) | mmap (ms) |
|---|---|---|---|---|---|
| 0.6B | 203 | 49 | 726 | 37 | 0.26 |
| 3B | 738 | 244 | 2389 | 239 | 0.82 |
| 4B | 953 | 313 | 1938 | 315 | 0.66 |
| 8B | 1904 | 551 | 2588 | 2135 | 0.72 |
| 14B | 3437 | 2740 | 4190 | 4890 | — |
| 32B | 9337 | 11085 | 10072 | 9513 | — |

### Perf Counters (Cold-Cache, 0.6B)
| Metric | sync | async | default | io_uring |
|---|---|---|---|---|
| Cycles | 5.88B | 3.71B | 5.14B | 3.69B |
| Instructions | 3.40B | 3.28B | 3.39B | 3.31B |
| IPC | 0.58 | 0.89 | 0.66 | 0.90 |
| L1 Miss % | 8.10% | 6.91% | 7.62% | 6.89% |
| Cache Miss % | 6.24% | 4.34% | 6.20% | 4.06% |
| Context Switches | 674 | 1087 | 977 | 35 |
| CPUs Used | 4.1x | 0.9x | 4.8x | 1.0x |
| Page Faults | 367K | 367K | 367K | 367K |

### Perf Counters (Cold-Cache, 14B io_uring)
| Metric | Value |
|---|---|
| Cycles | 67.6B |
| Instructions | 61.4B |
| IPC | 0.91 |
| L1 Miss % | 7.01% |
| Cache Miss % | 2.99% |
| Context Switches | 183 |
| CPUs Used | 1.0x |
| Page Faults | 7.2M |

### Perf Counters (Cold-Cache, 32B io_uring)
| Metric | Value |
|---|---|
| Cycles | 146.3B |
| Instructions | 130.0B |
| IPC | 0.89 |
| L1 Miss % | 7.37% |
| Cache Miss % | 2.87% |
| Context Switches | 323 |
| CPUs Used | 0.95x |
| Page Faults | 16.0M |

## Bottleneck Analysis

### 1. io_uring is consistently the slowest backend
- Across ALL model sizes, io_uring is 2-5x slower than sync
- Even at 32B, sync (11s) beats io_uring (48s) by 4x
- io_uring uses only 1 CPU core (0.95x), while sync uses 4+ cores
- Context switches are near-zero (35-323) confirming single-threaded execution
- strace shows 92.7% of time in `io_uring_enter` — the syscall overhead dominates

### 2. Why io_uring underperforms
- The current implementation processes tensors sequentially through io_uring
- No batching or parallel submission of read operations
- Each tensor gets its own io_uring submission/completion cycle
- For SafeTensors with hundreds of tensors, this means hundreds of io_uring_enter calls
- The setup overhead per operation exceeds the benefit of async I/O

### 3. Sync wins through parallelism
- Sync backend uses rayon thread pool for parallel shard reading
- 4+ CPU cores utilized vs 1 for io_uring
- Higher L1 miss rate (8.1% vs 6.9%) is the trade-off for parallelism
- But the wall-clock time is 4x faster because of CPU parallelism

### 4. Async is single-threaded but consistent
- Uses Tokio runtime but processes shards sequentially
- Lower L1/cache miss rates than sync (6.9% vs 8.1%)
- But wall-clock is 2x slower because no parallelism
- Becomes competitive at 8B-14B where sync's memory pressure increases

### 5. Default heuristic accuracy
- Correctly picks sync for 0.6B, 3B, 4B
- Incorrectly picks async for 8B (where sync is faster on cold load)
- Incorrectly picks async for 14B (where sync is faster on cold load)
- Picks sync for 32B (correct — async OOMs)

### 6. ServerlessLLM is 2-3x slower than SafeTensors
- Partition fragmentation kills parallelism
- Multiple file opens per partition
- Index parsing overhead
- sync is always fastest for ServerlessLLM

## Recommended Heuristic Changes

### SafeTensors Backend Chooser
Current logic uses shard_count and total_bytes only. Should add:
- For models < 6B: sync (parallel shard reading wins)
- For models 6B-14B: async (sync memory pressure increases)
- For models > 14B: sync (async OOM risk, sync parallelism still wins)
- mmap: always fastest for warm/repeated access

### io_uring Backend
The current io_uring implementation is not competitive. To make it useful:
- Batch multiple tensor reads into single io_uring submission
- Use io_uring's parallel submission queue
- Or remove io_uring as a backend option entirely

### ServerlessLLM Backend Chooser
- sync is always fastest — the partition layout prevents async parallelism benefits
- default should always pick sync for ServerlessLLm
