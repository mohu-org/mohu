# Roadmap

## Organization

The following repositories are planned for the `mohu-org` GitHub organization.

### Day one

| Repo | Purpose |
|---|---|
| [`mohu`](https://github.com/mohu-org/mohu) | Core Rust library + Python bindings — this repo |
| `.github` | Org profile (`README.md` on the org page), `FUNDING.yml` |
| `mohu-benchmarks` | Standalone benchmark suite: mohu vs NumPy vs JAX vs PyTorch on real workloads. Lives separately so anyone can run it without building the full library |
| `mohu-docs` | Documentation site (mdBook). API reference, guides, NumPy migration. Deployed to GitHub Pages |

### Before first public release

| Repo | Purpose |
|---|---|
| `mohu-compat` | NumPy compatibility layer — tracks which `numpy.*` API surface mohu covers, runs NumPy's own test suite against mohu, publishes a compatibility matrix |
| `mohu-examples` | Jupyter notebooks and scripts showing real use cases — NumPy replacement in data pipelines, scientific computing, ML preprocessing |

### As the ecosystem grows

| Repo | Purpose |
|---|---|
| `mohu-plugin` | Plugin/extension API for custom dtypes and ops |
| `mohu-arrow` | Arrow fork if divergence from arrow-rs becomes necessary (see Polars → arrow2 precedent) |

---

## Library (`mohu`)

### Foundation

- [ ] `mohu-error` — `MohuError` type, `MohuResult<T>` alias, PyO3 error conversions
- [ ] `mohu-core` — `NdArray<T>`, `DType`, `Shape`, `Buffer`, `Layout` (C/Fortran/custom strides), `SliceInfo`
- [ ] `mohu-compute` — SIMD kernels: reductions, element-wise ops, casts, bitmap validity

### Operations

- [ ] `mohu-ops` — broadcasting engine, arithmetic, comparison, logical, reduction ops
- [ ] `mohu-linalg` — matmul, LU, QR, SVD, Cholesky, norms, linear solvers, eigenvalues
- [ ] `mohu-stats` — descriptive stats, probability distributions, random sampling

### I/O

- [ ] `mohu-io` — `.npy`/`.npz`, CSV, Apache Arrow IPC, memory-mapped arrays

### Python

- [ ] `mohu-py` — PyO3 bindings, zero-copy NumPy buffer protocol, ABI3 wheels (`abi3-py310`), jemalloc
- [ ] Publish to PyPI

### Quality

- [ ] CI matrix: Python 3.10–3.13 + free-threaded 3.14t
- [ ] `cargo bench` baseline on core ops
- [ ] `cargo deny` dependency audit in CI
