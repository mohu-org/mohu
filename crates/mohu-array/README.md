## Goals

`mohu-array` aims to provide:

- A strongly typed multidimensional array API
- Efficient memory handling through `mohu-buffer`
- Ergonomic numerical computing primitives
- Extensible APIs for scientific and data-processing workloads

## Example (Planned)

```rust
let arr = NdArray::<f32>::zeros([2, 3]);

let reshaped = arr.reshape([3, 2]);

let transposed = reshaped.transpose();