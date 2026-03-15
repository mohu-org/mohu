# Contributing to mohu

## Prerequisites

- Rust stable (see `rust-toolchain.toml`)
- Python 3.10+ (for `mohu-py` development)

## Workflow

1. Fork and clone the repo
2. Create a branch: `git checkout -b feat/your-feature`
3. Make changes and verify: `make check`
4. Open a PR using the template

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(core): add strided slice iterator
fix(ops): correct broadcast shape inference for rank-0 arrays
perf(linalg): use BLAS sgemm for f32 matmul
```

Types: `feat`, `fix`, `perf`, `refactor`, `doc`, `test`, `chore`, `ci`

Breaking changes: append `!` after the type — `feat(core)!: rename Array → NdArray`

## Crate responsibilities

| Crate | Owns |
|---|---|
| `mohu-core` | `NdArray`, `DType`, `Shape`, `Buffer`, `Layout`, `SliceInfo` |
| `mohu-ops` | Broadcasting, element-wise arithmetic/cmp/logical, reductions |
| `mohu-linalg` | matmul, LU, QR, SVD, Cholesky, norms, solvers, eigenvalues |
| `mohu-stats` | Descriptive stats, distributions, random sampling |
| `mohu-io` | `.npy`/`.npz`, CSV, Arrow IPC, memory-mapped arrays |
| `mohu-py` | Python module, PyO3 bindings, NumPy buffer protocol |

## Running tests

```sh
make test      # all workspace tests
make lint      # clippy (warnings = errors)
make bench     # benchmarks
```
