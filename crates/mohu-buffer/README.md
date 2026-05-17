# mohu-buffer

Memory foundation of the [mohu](https://github.com/mohu-org/mohu) scientific computing library. This crate provides reference-counted, strided, typed byte storage (`Buffer`) with layout arithmetic, parallel element-wise ops, allocation pooling, and DLPack v0.8 interop. Every ndarray in `mohu-array` is backed by a `Buffer` from this crate.

## Architecture

```
Buffer = Arc<RawBuffer> + DType + Layout + BufferFlags
           │
           └── RawBuffer: aligned bytes + AllocHandle
Layout: shape + strides + byte offset (views, reshape, transpose)
```

## Modules

| Module | Responsibility |
|--------|----------------|
| [`alloc`](src/alloc.rs) | Aligned heap / mmap allocation, live-byte stats |
| [`strides`](src/strides.rs) | Stride computation, N-dim index iteration, broadcast |
| [`layout`](src/layout.rs) | Shape + stride + offset descriptor, view operations |
| [`buffer`](src/buffer.rs) | Reference-counted typed buffer; DLPack import/export |
| [`view`](src/view.rs) | Typed lifetime-bound views (`BufferView<T>`) |
| [`ops`](src/ops.rs) | Parallel fill, copy, cast (Rayon) |
| [`pool`](src/pool.rs) | Allocation reuse pool (`GLOBAL_POOL`) |

## Example

```rust
use mohu_buffer::Buffer;
use mohu_dtype::DType;

let mut buf = Buffer::zeros(DType::F64, &[2, 3])?;
buf.fill(1.0)?;

let from_data = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?
    .reshape(&[2, 3])?;

let view = from_data.transpose();
let slice = view.as_slice::<f64>()?;
assert_eq!(slice.len(), 6);
```

## DLPack

- [`Buffer::to_dlpack`](https://docs.rs/mohu-buffer/latest/mohu_buffer/struct.Buffer.html#method.to_dlpack) exports a `DLManagedTensor` pointer; the consumer must call the embedded deleter when finished.
- [`Buffer::from_dlpack`](https://docs.rs/mohu-buffer/latest/mohu_buffer/struct.Buffer.html#method.from_dlpack) imports foreign tensors; the caller must ensure the tensor outlives any borrowed views or clone into an owned `Buffer`.

Only CPU device tensors are supported in the current release.

## Copy-on-write

- [`share`](https://docs.rs/mohu-buffer/latest/mohu_buffer/struct.Buffer.html#method.share) returns a new `Buffer` handle pointing at the same allocation (cheap, read-only sharing).
- [`make_unique`](https://docs.rs/mohu-buffer/latest/mohu_buffer/struct.Buffer.html#method.make_unique) clones underlying storage when the reference count is greater than one, so in-place writes are safe.

## Contributing

See the workspace [CONTRIBUTING.md](../../CONTRIBUTING.md).
