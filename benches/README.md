# Benchmarks

This directory contains Criterion-based benchmarks for mohu's performance-critical code paths.

## Current scope

`mohu-array` (the public `NdArray<T>` type) is not yet implemented, so these
benchmarks currently exercise the `mohu-buffer` allocation layer that
`NdArray` will be built on top of. Once `NdArray` lands, these benchmarks
should be extended to cover array-level operations (creation, indexing,
elementwise arithmetic, reductions, etc).

## Running locally

```bash
cargo bench -p mohu-buffer --bench ndarray_ops
```

To save your own named baseline for comparison:

```bash
cargo bench -p mohu-buffer --bench ndarray_ops -- --save-baseline my-branch
```

## Comparing against the committed baseline

Install [critcmp](https://github.com/BurntSushi/critcmp):

```bash
cargo install critcmp
```

Then compare your local run against the committed baseline:

```bash
critcmp benches/baselines/main.json my-branch
```

## Current baseline

Captured on a GitHub Codespaces cloud runner (shared vCPU, not a dedicated
benchmark machine). Absolute numbers will vary across hardware — treat this
as a relative regression baseline, not an authoritative performance claim.

| Benchmark                  | Time (median) |
|-----------------------------|---------------|
| buffer_zeros_f64/100        | ~245 ns       |
| buffer_zeros_f64/10000      | ~1.27 µs      |
| buffer_zeros_f64/1000000    | ~6.9 µs       |
| buffer_alloc_f64/100        | ~217 ns       |
| buffer_alloc_f64/10000      | ~277 ns       |
| buffer_alloc_f64/1000000    | ~6.5 µs       |

Full raw data: `benches/baselines/main.json`

## Intended NumPy comparison targets (future work)

Once `NdArray` is implemented, the intent is to benchmark these operations
against their NumPy equivalents (by name, run separately in Python — this
repo does not execute NumPy in CI):

| mohu operation        | NumPy equivalent         |
|------------------------|---------------------------|
| `NdArray::zeros`       | `numpy.zeros`             |
| `NdArray::from_vec`    | `numpy.array`             |
| elementwise add/mul    | `numpy.add` / `numpy.multiply` |
| `sum` / `mean` reduce  | `numpy.sum` / `numpy.mean`|

## CI regression gate

On every pull request, CI runs these benchmarks and compares them against
`benches/baselines/main.json` using critcmp, flagging any benchmark that
regresses by more than 5% for human review. See
`.github/workflows/bench-check.yml`.