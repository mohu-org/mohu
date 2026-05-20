# mohu-array

`mohu-array` provides `NdArray<T>`, the core typed N-dimensional array type in the [mohu](../../README.md) workspace. It composes `mohu-buffer` (memory + layout), `mohu-dtype` (scalar type system), and shape/slice/view abstractions into a single ergonomic Rust generic interface for numerical computing.

---

## Relationship to `mohu-buffer`

```
NdArray<T>  =  mohu-buffer::Buffer  +  type parameter T : Scalar (from mohu-dtype)
```

`mohu-buffer` owns the raw bytes, shape, and strides. `mohu-array` adds the generic type parameter `T` (bounded by `num-traits` scalar traits and `mohu-dtype`'s type system) so that element-level operations are type-safe without runtime casting. Parallel operations are powered by `rayon`.

Internal modules:

| Module | Role |
|---|---|
| `array` | `NdArray<T>` struct definition |
| `shape` | Shape representation and validation |
| `slice` | Slice argument types for sub-array selection |
| `view` | Zero-copy array views |
| `iter` | Iterators over elements and axes |

---

## Planned API Surface

> **Status:** implementation is in progress — all source files are currently stubs.
> The API below reflects the planned design; signatures may change before stabilisation.
> See the [issue tracker](https://github.com/mohu-org/mohu/issues?q=label%3A%22crate%3A+mohu-array%22) for progress.

### Construction

```rust
// All-zeros array of shape [3, 4]
let a = NdArray::<f64>::zeros(&[3, 4]);

// All-ones array
let b = NdArray::<f32>::ones(&[2, 2, 2]);

// From an existing flat slice (row-major by default)
let c = NdArray::<i32>::from_slice(&[1, 2, 3, 4, 5, 6], &[2, 3]);
```

### Indexing

```rust
// Element access
let val: f64 = a[[1, 2]];

// Mutable element access
a[[0, 0]] = 3.14;

// Slicing — returns a zero-copy view
let row = a.slice(s![0, ..]);
```

### Arithmetic

```rust
// Element-wise operators (parallel via rayon internally)
let sum  = &a + &b;
let diff = &a - &b;
let prod = &a * &b;

// Scalar broadcast
let scaled = &a * 2.0;
```

### Shape Manipulation

```rust
// Reshape (total element count must match)
let flat = a.reshape(&[12]);

// Transpose — reverses axis order, returns a view
let t = a.transpose();
```

---

## Status

`mohu-array` is **under active development** and is not yet published to [crates.io](https://crates.io). All `.rs` files are currently module stubs. Implementation is being tracked in the [mohu issue tracker](https://github.com/mohu-org/mohu/issues?q=label%3A%22crate%3A+mohu-array%22).

Do not depend on this crate in production — public API stability is not guaranteed before the first versioned release.

---

## Related Crates

| Crate | Role |
|---|---|
| [`mohu-buffer`](../mohu-buffer/README.md) | Untyped memory layer — raw bytes, shape, strides |
| `mohu-dtype` | `Scalar` trait and `DType` enum used by `NdArray<T>` |
| `mohu-error` | Structured error types used across the workspace |
| `mohu-core` | Re-export facade — import this instead of individual crates |

---

## Contributing

Please read the workspace-level [CONTRIBUTING.md](../../CONTRIBUTING.md) before opening a pull request. All contributions — bug reports, API design feedback, and code — are welcome.
