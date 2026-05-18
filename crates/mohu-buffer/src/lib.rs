/// mohu-buffer — raw buffer allocation, memory layout, and stride arithmetic.
///
/// This crate is the memory foundation of mohu.  Every ndarray in `mohu-array`
/// ultimately holds a `Buffer` from this crate.  The public API is structured
/// in layers:
///
/// | Module      | Responsibility                                          |
/// |-------------|----------------------------------------------------------|
/// | [`alloc`]   | Aligned heap / mmap allocation, global live-byte stats   |
/// | [`strides`] | Stride computation, N-dim index iteration, broadcast     |
/// | [`layout`]  | Shape + stride + offset descriptor, all view operations  |
/// | [`buffer`]  | Reference-counted typed buffer; DLPack import/export     |
/// | [`view`]    | Typed lifetime-bound views (`BufferView<T>`)              |
/// | [`ops`]     | Parallel fill, copy, cast using Rayon                    |
/// | [`pool`]    | Allocation reuse pool; `GLOBAL_POOL` singleton           |
pub mod alloc;
pub mod buffer;
pub mod layout;
pub mod ops;
pub mod pool;
pub mod strides;
pub mod view;

// ─── Re-exports ───────────────────────────────────────────────────────────────

pub use alloc::{AllocHandle, AllocStats, CACHE_LINE, MMAP_THRESHOLD, SIMD_ALIGN, Strategy};

pub use buffer::{
    Buffer, BufferFlags, DLManagedTensor, DLTensor, RawBuffer, RawDLDataType, RawDLDevice,
};

pub use layout::{Layout, Order, SliceArg};

pub use ops::{
    cast_copy, copy_to_contiguous, fill, fill_one, fill_raw, fill_zero, parallel_inplace,
    parallel_map, reduce,
};

pub use pool::{BufferPool, GLOBAL_POOL, PoolStats};

pub use strides::{
    NdIndexIter, ShapeVec, StrideVec, StridedByteIter, broadcast_strides, byte_offset, c_strides,
    contiguous_nbytes, f_strides, ravel_multi_index, shape_size, unravel_index, validate_strides,
};

pub use view::{BufferView, BufferViewMut};

// Re-export error types for convenience.
pub use mohu_error::{MohuError, MohuResult};
