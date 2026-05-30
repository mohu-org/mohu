# Mohu Benchmarks

This crate provides a Criterion.rs-based benchmark scaffold for the Mohu project. It intentionally starts small with well-documented microbenchmarks for:

- Element-wise arithmetic (vector add)
- Reduction (sum)
- Memory stride access comparison (row-major vs column-major scanning)

Maintainers and contributors should extend this suite with:

- `mohu-array` focused benchmarks for `compute`, `ufunc`, `ops` paths
- I/O benchmarks for `.npy`/.csv` using `mohu-io`
- Cross-comparisons with NumPy (scripted externally via Python)
- SIMD vs scalar kernels and threaded vs sequential experiments

Run locally:

```sh
cd crates/mohu-benchmarks
cargo bench --bench bench_core
```

CI integration recommendation: run benchmarks in a scheduled job, store the resulting Criterion reports/artifacts in an S3-compatible storage, and detect regressions by comparing specific baseline outputs or trend analysis.
