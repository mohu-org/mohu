# MOHU College Demo

## Run

From the repository root:

```bash
cargo run --release --example college_demo
```

Run twice from a clean terminal. The command has no network or external-data dependency.

## What the demo proves

1. **Array construction** — `NdArray<f64>` accepts typed row-major data and exposes values, shape, rank, and runtime dtype.
2. **Views and strides** — reshape and transpose preserve logical values; transpose shares backing storage and changes strides instead of copying.
3. **Broadcasting** — a `[3]` row becomes a `[2, 3]` read-only view with a zero stride on the repeated axis.
4. **Matrix multiplication** — `mohu-ops::matmul` computes the documented 2×2 result through the public `Buffer` semantic API.
5. **Seeded randomness** — equal seeds produce equal integer buffers.

## 30-second explanation

“MOHU already has a typed Rust array layer backed by a reference-counted buffer. Metadata is explicit, layout transformations are views, broadcasting uses zero strides, matrix multiplication has a tested public semantic owner in `mohu-ops`, and seeded generation is reproducible. These are narrow working foundations, not claims of NumPy completeness.”

## Validation

The shown behavior is covered by existing `mohu-array`, `mohu-buffer`, `mohu-ops`, and `mohu-random` tests. The executable also asserts the matrix result and seeded reproducibility.

## Prototype limits

- Rust demo is the guaranteed path.
- FFT currently supports complex C64/C128 `fft`/`ifft`; real and multidimensional helpers remain incomplete.
- Python bindings are a separate prototype and are not required for this backup demo.
- No performance, NumPy compatibility, or full API-completeness claim is made.
