# Project Development Diary

## 2025-09-30

- Scaffolded initial Zig project structure with build.zig and build.zig.zon files
- Added ServerlessLLM repo to use as a reference to my own implementation
- Migrated from Zig to Rust due to WriterGate breaking changes in Zig I/O system
- Removed Zig build files and source directory
- Updated proposal.md to use Rust instead of Zig
- Completed comprehensive research phase covering ServerlessLLM architecture analysis
- Researched io_uring ecosystem - found Tokio has no built-in support, tokio-uring crate available
- Designed TensorStore format specification with 64-byte alignment to solve safetensors GPU issues
- Created research documentation structure in /research/ with 5 detailed analysis documents
- Updated roadmap.md to reflect completed research phase and MVP-focused next steps
- Clarified project scope: custom TensorStore format is for learning, real innovation is io_uring loading approach
- Updated MVP to focus on safetensors + tokio-uring vs safetensors + tokio::fs comparison
- Redesigned project scope for 300-hour constraint
- Refocused roadmap on validation rather than production system
- Updated implementation strategy to prioritize empirical testing over features
- Established Week 6 go/no-go decision point for performance validation
- Revised time estimates to be more realistic for competent engineer (300 hours → MVP in weeks 3-4, analysis weeks 5-7)
