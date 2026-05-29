# Roadmap

> **See also**: [README.md](./README.md) for project status, [CRATE_MAP.md](./CRATE_MAP.md) for implementation status per crate.

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

---

## Ecosystem Integration

The goal is that any library expecting a NumPy array or a PyTorch tensor can consume a mohu array **without copying and without knowing mohu exists**. This is done through a stack of standard protocols, implemented in `mohu-py` on top of layout/pointer info exposed by `mohu-buffer`.

### Tier 1 — NumPy and the old ecosystem

| Protocol | What it unlocks | Owner |
|---|---|---|
| Python buffer protocol (`__buffer__`) | Zero-copy `np.asarray(x)`, PIL, Pandas, anything that calls `PyBUF_SIMPLE` | `mohu-py` + `mohu-buffer` |
| `__array__()` | Fallback for older libraries that call `np.asarray(x)` before checking buffer protocol | `mohu-py` |
| `__array_interface__` | NumPy structured zero-copy view (shape, strides, typestr, data pointer) | `mohu-py` |

Once Tier 1 is done: `sklearn.fit(mohu_array)`, `pd.DataFrame(mohu_array)`, `PIL.Image.fromarray(mohu_array)` all work.

### Tier 2 — ML ecosystem (PyTorch, JAX, CuPy, TensorFlow)

| Protocol | What it unlocks | Owner |
|---|---|---|
| `__dlpack__()` + `__dlpack_device__()` | Zero-copy tensor exchange with PyTorch, JAX, CuPy, TF, MXNet — expose a `DLManagedTensor` pointer from Rust, the framework consumes it directly, no copies | `mohu-py` + `mohu-buffer` |

**This is the answer to "can I use a mohu array directly in a PyTorch op".** Once DLPack is implemented:

```python
import mohu as mu
import torch

x = mu.array([1.0, 2.0, 3.0])
t = torch.from_dlpack(x)   # zero-copy, no data moved
t = t * 2                  # runs on PyTorch
```

Same pattern works for JAX (`jax.dlpack.from_dlpack`), CuPy, TensorFlow. The mohu array never gets converted — the framework reads mohu's memory directly.

### Tier 3 — Drop-in replacement

| Protocol | What it unlocks | Owner |
|---|---|---|
| Array API Standard (`__array_namespace__()`) | scikit-learn (since 1.0), scipy, and any Array-API-aware library dispatches to mohu natively — no conversion needed | `mohu-py` |
| `__array_ufunc__()` | NumPy ufuncs (`np.sin(x)`, `np.add(x, y)`) dispatch to mohu's implementation instead of converting | `mohu-py` |
| `__array_function__()` | NumPy functions (`np.stack`, `np.concatenate`, `np.dot`) dispatch to mohu — the SKIP protocol | `mohu-py` |

Once Tier 3 is done, `np.sin(mohu_array)` calls mohu's sin, not NumPy's. scikit-learn pipelines work on mohu arrays end to end.

### Implementation order

```
mohu-buffer exposes: raw data pointer, shape, strides, dtype, device
         ↓
Tier 1: buffer protocol + __array__          ← scikit-learn, pandas, PIL
         ↓
Tier 2: DLPack                               ← PyTorch, JAX, CuPy, TensorFlow
         ↓
Tier 3: Array API Standard                   ← sklearn native, scipy
         ↓
Tier 3: __array_ufunc__ + __array_function__ ← full NumPy drop-in
```

### Quality

- [ ] CI matrix: Python 3.10–3.13 + free-threaded 3.14t
- [ ] `cargo bench` baseline on core ops
- [ ] `cargo deny` dependency audit in CI
