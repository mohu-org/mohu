# Mohu Benchmarks

This crate provides a Criterion.rs-based benchmark scaffold for the Mohu project. It intentionally starts small with well-documented microbenchmarks for:

- Element-wise arithmetic (vector add)
- Reduction (sum)
- Memory stride access comparison (row-major vs column-major scanning)

## Architecture Note

**Important**: The ROADMAP.md suggests `mohu-benchmarks` should be a standalone repository to allow anyone to run benchmarks without building the full library. However, this PR implements it as a workspace member crate for tighter integration with the CI pipeline and easier maintenance. If maintainers prefer the standalone repo approach, this can be migrated in a follow-up PR.

## Current Status

**Note**: `mohu-array` is currently in stub status (module files are empty as documented in CRATE_MAP.md). The current benchmarks use plain `Vec<T>` operations as placeholders. Once `mohu-array` is implemented, these benchmarks should be extended to use the actual array API.

### What's Implemented

- Basic microbenchmarks using `Vec<T>` operations
- Integration test placeholder in `tests/` directory
- CI workflow for scheduled benchmark runs (`.github/workflows/benchmark.yml`)
- Proper use of `criterion::black_box()` to prevent compiler optimization

### Next Steps

Once `mohu-array` is implemented, maintainers and contributors should extend this suite with:

1. **Integration Tests**: Replace the placeholder in `tests/integration_test.rs` with actual `mohu-array` construction and operation tests
2. **Array Benchmarks**: Extend benchmarks to use `mohu-array` compute paths (ufunc/reduction APIs directly in bench closures)
3. **I/O Benchmarks**: Add benchmarks for `.npy`/`.csv` using `mohu-io`
4. **Cross-comparisons**: Add NumPy comparisons (scripted externally via Python)
5. **Advanced Experiments**: SIMD vs scalar kernels and threaded vs sequential experiments
6. **Baseline Reports**: Generate and commit baseline `target/criterion/` HTML reports or JSON summaries for regression detection (requires properly configured build environment with MSVC toolchain on Windows)

## Running Benchmarks

Run locally:

```sh
cd crates/mohu-benchmarks
cargo bench --bench bench_core
```

Run with bencher output format (for CI):

```sh
cd crates/mohu-benchmarks
cargo bench -- --output-format bencher | tee output.txt
```

## CI Integration

A scheduled benchmark workflow is configured in `.github/workflows/benchmark.yml`:
- Runs every Monday at 3:00 UTC via cron schedule
- Can also be triggered manually via `workflow_dispatch`
- Uploads benchmark results and Criterion reports as artifacts (30-day retention)
- Uses `bencher` output format for potential integration with `github-action-benchmark` for trend charts

For full regression detection, consider:
- Storing Criterion reports in S3-compatible storage
- Using `github-action-benchmark` for trend visualization
- Comparing specific baseline outputs across runs
