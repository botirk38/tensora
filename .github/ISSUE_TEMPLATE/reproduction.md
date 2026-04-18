---
name: Paper / measurement reproduction
about: Ordering or timings differ from the paper when following README procedures
title: "[repro] "
labels: question
---

## What you are trying to reproduce

- Table or claim (e.g. Rust SafeTensors ordering, anchor reps):
- Script or command (from repository root `README.md`):

## Your commit and host

- `git rev-parse HEAD`:
- Storage (local NVMe / cloud / other):
- Kernel `uname -r`:

## What you observed

- Ordering (which backend was fastest?):
- Rough timings (optional):

## What you expected (from the paper)

## Notes

Cold-cache procedure (`sync` + `drop_caches`) must match the paper; replication targets **ordering**, not identical milliseconds.
