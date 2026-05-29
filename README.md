# mohu

Rust-powered arrays for Python. Fast, parallel, and built for the future.

[![CI](https://github.com/mohu-org/mohu/actions/workflows/ci.yml/badge.svg)](https://github.com/mohu-org/mohu/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![MSRV](https://img.shields.io/badge/rust-1.85%2B-blue)](./rust-toolchain.toml)

mohu is an early-stage NumPy replacement with its core written in Rust. The goal is simple — take everything Python's scientific stack does and do it without the bottlenecks that have been accepted for decades.

No GIL. No single-threaded ops. No object overhead. Just arrays.

## Why

NumPy is written in C and hasn't fundamentally changed in 20 years. It's single-threaded by default, its string arrays are an afterthought, and parallelism requires reaching for other tools. The Python data ecosystem deserves a better foundation.

Polars proved you can rewrite the data layer in Rust and win. mohu is that same bet, one layer down.

## Project Status

Early-stage. The foundation crates (`mohu-error`, `mohu-dtype`, `mohu-buffer`) are fully implemented. Higher-level compute, sparse, and I/O crates are stubbed and planned. See [ROADMAP.md](./ROADMAP.md) for the full plan.

## Project Structure

```
mohu/
├── Cargo.toml          # Workspace root — all dep versions here
├── crates/             # 17 workspace crates
│   ├── mohu-error/     # Error types, codes, reporter (fully implemented)
│   ├── mohu-dtype/     # DType enum, scalar traits, type promotion (fully implemented)
│   ├── mohu-buffer/    # Raw allocation, layout, strides, DLPack (fully implemented)
│   ├── mohu-array/     # NdArray<T> type (stub)
│   ├── mohu-core/      # Re-export facade (implemented)
│   ├── mohu-simd/      # SIMD kernels: AVX2, AVX-512, NEON (stub)
│   ├── mohu-ufunc/     # Universal-function protocol (stub)
│   ├── mohu-index/     # Advanced indexing (stub)
│   ├── mohu-ops/       # Element-wise ops, broadcasting (stub)
│   ├── mohu-fft/       # FFT transforms (partially implemented)
│   ├── mohu-random/    # PRNG + statistical distributions (partially implemented)
│   ├── mohu-special/   # Special math functions (stub)
│   ├── mohu-stats/     # Descriptive statistics (stub)
│   ├── mohu-sparse/    # COO/CSR/CSC sparse formats (stub)
│   ├── mohu-masked/    # Masked arrays (stub)
│   ├── mohu-io/        # File I/O: .npy, .npz, CSV, Arrow (stub)
│   └── mohu-testing/   # Test utilities (stub)
├── benches/            # Criterion benchmarks
├── docs/               # mdBook source, design docs (ADRs), and RFCs
├── examples/           # Usage examples
├── scripts/            # Build and CI helper scripts
├── tests/              # Integration tests
└── .github/            # CI workflows, issue templates, auto-assignment
```

See [CRATE_MAP.md](./CRATE_MAP.md) for the full public API surface per crate.

## Architecture

The workspace is organized into strict dependency layers — crates may only depend on layers below them:

| Layer | Crates | Purpose |
|-------|--------|---------|
| Foundation | error, dtype, buffer, array, core | Shared types, allocation, array representation |
| Dispatch | simd, ufunc, index | SIMD kernels, universal functions, indexing protocols |
| Compute | ops, fft, random, special, stats | Numerical operations and transforms |
| Extensions | sparse, masked | Specialized data structures |
| IO/Tooling | io, testing | Serialization and developer tooling |

## What's Coming

- N-dimensional arrays with a NumPy-compatible API
- Parallel operations by default via Rayon
- First-class string arrays — not `dtype=object`
- Built on Apache Arrow — interop with Polars, DuckDB, and the rest of the ecosystem out of the box
- Zero-copy Python integration via PyO3
- SIMD-accelerated math ops
- Memory layouts NumPy can't express

## Quick Start

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| Rust | 1.85 (edition 2024) |
| Python | 3.10+ (for `mohu-py` bindings) |

### Build

```sh
cargo build --workspace --all-features
```

### Test

```sh
cargo test --workspace --all-features
```

### Lint

```sh
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the full development workflow.

## Built With

- [Rust](https://rust-lang.org)
- [PyO3](https://github.com/PyO3/pyo3) — Python bindings
- [arrow-rs](https://github.com/apache/arrow-rs) — columnar memory format
- [Rayon](https://github.com/rayon-rs/rayon) — data parallelism

## Documentation

- [Introduction](./docs/src/introduction.md) — What mohu is and why it exists
- [Architecture](./docs/design/) — Design docs (ADRs) for major decisions
- [RFCs](./docs/rfcs/) — Proposals for significant changes
- [API Reference](https://docs.rs/mohu-core/) — Rustdoc (when published)

### Key Documents

| Document | Purpose |
|----------|---------|
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Development workflow, commit conventions, CI checks |
| [CRATE_MAP.md](./CRATE_MAP.md) | Complete crate-by-crate public API surface |
| [ROADMAP.md](./ROADMAP.md) | Project milestones and planned features |
| [CHANGELOG.md](./CHANGELOG.md) | Release history |
| [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md) | Community guidelines |
| [SECURITY.md](./SECURITY.md) | Vulnerability reporting |
| [AGENTS.md](./AGENTS.md) | Guidance for AI coding agents |
| [CLAUDE.md](./CLAUDE.md) | AI assistant guide for the workspace |

## Contributing

Contributions are welcome. Please read:

1. [CONTRIBUTING.md](./CONTRIBUTING.md) — full workflow, commit conventions, and CI checks
2. [CRATE_MAP.md](./CRATE_MAP.md) — understand crate ownership before making changes
3. [AGENTS.md](./AGENTS.md) — guidance for AI agents (if using one)

Every commit must be signed off (`git commit -s`) in compliance with the [Developer Certificate of Origin](https://developercertificate.org/).

## License

MIT — see [LICENSE](./LICENSE).
