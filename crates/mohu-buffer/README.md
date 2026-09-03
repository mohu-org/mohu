# mohu-buffer

Raw buffer allocation, memory layout, and stride arithmetic — the memory foundation of
[mohu](https://github.com/mohu-org/mohu). Every ndarray in `mohu-array` ultimately holds
a `Buffer` from this crate.

The central idea is that **layout is separate from storage**. `Layout` owns shape,
strides, and offset; `Buffer` owns bytes and a `DType` tag. Transpose, reshape, slice,
broadcast, squeeze, and `expand_dims` are layout rewrites — they allocate nothing and
copy nothing.

## Layers

| Module | Responsibility |
|---|---|
| `alloc` | Aligned heap and mmap allocation, global live-byte stats |
| `strides` | Stride computation, N-dim index iteration, broadcast rules |
| `layout` | Shape + stride + offset descriptor, all view operations |
| `buffer` | Reference-counted typed buffer, DLPack import and export |
| `view` | Typed lifetime-bound views — `BufferView<T>`, `BufferViewMut<T>` |
| `ops` | Parallel fill, copy, cast, scan, gather/scatter via Rayon |
| `pool` | Allocation reuse pool and the `GLOBAL_POOL` singleton |

## Usage

```rust
use mohu_buffer::{Buffer, Order};
use mohu_dtype::{CastMode, DType};

// Build a 2x3 buffer from a typed slice. dtype is inferred from T.
let a = Buffer::from_slice_2d::<f32>(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?;

// Views are zero-copy: transpose only rewrites the stride vector.
let t = a.transpose();
assert_eq!(t.shape(), &[3, 2]);
assert_eq!(t.get::<f32>(&[2, 1])?, 6.0);

// Constructors mirror NumPy.
let r = Buffer::arange(0.0, 10.0, 2.0, DType::I32)?;
assert_eq!(r.to_vec::<i32>()?, vec![0, 2, 4, 6, 8]);

// Broadcasting produces a zero-copy view with 0-strides.
let row = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0])?;
assert_eq!(row.broadcast_to(&[2, 3])?.strides(), &[0, 4]);

// Fortran order is a first-class option, not an afterthought.
assert!(Buffer::alloc(DType::F32, &[2, 3], Order::F)?.is_f_contiguous());
# Ok::<(), mohu_error::MohuError>(())
```

## Allocation

Allocations are 64-byte aligned (`SIMD_ALIGN`) so kernels can assume aligned loads.
Buffers above `MMAP_THRESHOLD` (1 MiB) switch from the heap to `mmap` automatically;
`Strategy` lets a caller force either path. `AllocStats` tracks live bytes globally, and
`BufferPool` recycles freed allocations by size class to keep steady-state workloads off
the system allocator.

## Interop

`Buffer::to_dlpack()` exports a `DLManagedTensor` for zero-copy handoff to PyTorch,
JAX, CuPy, and anything else speaking DLPack. Imported buffers are marked external and
are not freed by mohu.

## Sharing and copy-on-write

`Buffer::share()` bumps a refcount rather than copying. `make_unique()` performs the
copy only when a writer needs exclusive ownership, and `is_shared()` reports whether a
buffer currently has other owners.

## Features

| Feature | Default | Effect |
|---|---|---|
| `mmap` | yes | Enables `memmap2`-backed allocation for large buffers |

## Examples

```bash
cargo run --example quickstart       # the workspace README quickstart
cargo run --example buffer_basics    # construction, views, reshape, slicing
cargo run --example alloc_and_pool   # allocation strategies and the buffer pool
```

## License

MIT — see [LICENSE](../../LICENSE).
