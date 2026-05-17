# mohu-buffer

**Raw buffer allocation, memory layout, and stride arithmetic for mohu arrays.**

This crate is the memory foundation of mohu. Every ndarray in `mohu-array` ultimately holds a `Buffer` from this crate. It provides typed, reference-counted, strided byte storage with zero-copy views and DLPack interoperability.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Buffer                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Arc<RawBuffer>  +  DType  +  Layout  +  BufferFlags        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  RawBuffer ──► AllocHandle (owned) or DLManagedTensor (external)    │
│                                                                      │
│  Layout ──► Shape + Strides + Offset + Itemsize                     │
│            (describes how to index into the backing bytes)          │
└─────────────────────────────────────────────────────────────────────┘
```

**Key relationships:**
- `Buffer` = `Arc<RawBuffer>` + `DType` + `Layout` + `BufferFlags`
- Multiple `Buffer` instances can share the same `RawBuffer` (slices, transposes, broadcasts)
- `Layout` does not own bytes — it only describes indexing
- Copy-on-write semantics: `share()` for shared views, `make_unique()` for mutable copies

## Modules

| Module      | Responsibility                                          |
|-------------|----------------------------------------------------------|
| [`alloc`]   | Aligned heap / mmap allocation, global live-byte stats   |
| [`strides`] | Stride computation, N-dim index iteration, broadcast     |
| [`layout`]  | Shape + stride + offset descriptor, all view operations  |
| [`buffer`]  | Reference-counted typed buffer; DLPack import/export     |
| [`view`]    | Typed lifetime-bound views (`BufferView<T>`)             |
| [`ops`]     | Parallel fill, copy, cast using Rayon                    |
| [`pool`]    | Allocation reuse pool; `GLOBAL_POOL` singleton           |

## Examples

### Creating buffers

```rust
use mohu_buffer::{Buffer, Layout};
use mohu_dtype::dtype::DType;

// Create a zero-filled buffer
let buf = Buffer::zeros(&[3, 4], DType::Float32)?;

// Create from a slice
let data = vec![1.0f32, 2.0, 3.0, 4.0];
let buf = Buffer::from_slice(&data, &[2, 2], DType::Float32)?;
```

### Reshaping and transposing

```rust
// Reshape (shares the same underlying bytes)
let reshaped = buf.reshape(&[4, 3])?;

// Transpose (creates a new Layout, same RawBuffer)
let transposed = buf.transpose()?;
```

### Accessing data

```rust
// Get a typed view
let view = buf.as_view::<f32>()?;

// Access as slice (only for contiguous buffers)
if buf.is_c_contiguous() {
    let slice = buf.as_slice::<f32>()?;
}
```

## DLPack Interoperability

DLPack is a cross-framework memory sharing standard. `mohu-buffer` supports zero-copy import/export:

### Import from DLPack

```rust
// Wrap externally-owned memory (e.g., from PyTorch, TensorFlow)
let buf = Buffer::from_dlpack(dl_managed_tensor)?;
// The deleter is called when the last Arc reference is dropped
```

**Safety contract:**
- The caller must ensure the `DLManagedTensor` is valid and not aliased
- The backing memory must remain valid for the lifetime of the `Buffer`
- Only CPU devices are currently supported

### Export to DLPack

```rust
// Export for consumption by other frameworks
let dl_tensor = buf.to_dlpack()?;
// The Buffer is kept alive until the consumer calls the deleter
```

## Copy-on-Write Semantics

`Buffer` uses `Arc<RawBuffer>` for shared ownership. Operations like slicing, reshaping, and transposing create new `Buffer` instances that share the same bytes:

```rust
// share() creates a view (no copy)
let view = buf.share();

// make_unique() creates an independent copy
let owned = buf.make_unique()?;  // now safe to mutate
```

**When to use:**
- `share()` — for read-only views, broadcasting, transposing
- `make_unique()` — before in-place mutation, or to decouple from external memory

## Features

- `default` — includes `mmap` feature
- `mmap` — memory-mapped allocation for large buffers (uses `memmap2`)

## License

MIT License. See [LICENSE](../../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.