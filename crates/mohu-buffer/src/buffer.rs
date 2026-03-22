/// The core buffer type: reference-counted, typed, strided byte storage.
///
/// `Buffer` = `Arc<RawBuffer>` + `DType` + `Layout` + `BufferFlags`.
///
/// Multiple `Buffer` instances can share the same underlying bytes (slices,
/// transposes, broadcasts) through Arc reference counting with copy-on-write
/// semantics for mutation.
///
/// # External memory (DLPack)
///
/// `Buffer` can wrap externally-owned memory via [`Buffer::from_dlpack`].
/// In that case the backing `RawBuffer` holds a `*mut DLManagedTensor`
/// and calls its deleter when the last Arc reference is dropped.
use std::{
    ptr::NonNull,
    sync::Arc,
};

use mohu_dtype::{
    dlpack::{assert_cpu_device, DLDataType},
    dtype::DType,
    promote::CastMode,
    scalar::Scalar,
};
use mohu_error::{MohuError, MohuResult};

use crate::{
    alloc::{AllocHandle, SIMD_ALIGN},
    layout::{Layout, Order, SliceArg},
    strides::contiguous_nbytes,
};

// ─── BufferFlags ─────────────────────────────────────────────────────────────

/// Bitfield flags describing the mutable/layout properties of a `Buffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferFlags(u8);

impl BufferFlags {
    /// Array is writeable (not read-only / not a broadcast view).
    pub const WRITEABLE:     Self = Self(1 << 0);
    /// This `Buffer` is the (sole or shared) owner of the backing bytes.
    pub const OWNS_DATA:     Self = Self(1 << 1);
    /// Array is C-contiguous in the backing buffer.
    pub const C_CONTIGUOUS:  Self = Self(1 << 2);
    /// Array is Fortran-contiguous in the backing buffer.
    pub const F_CONTIGUOUS:  Self = Self(1 << 3);
    /// Backing memory is SIMD-aligned (≥ 64 bytes).
    pub const ALIGNED:       Self = Self(1 << 4);

    /// Returns an empty flag set with no flags enabled.
    pub const fn empty() -> Self { Self(0) }
    /// Returns `true` if all flags in `other` are set in `self`.
    pub const fn contains(self, other: Self) -> bool { self.0 & other.0 == other.0 }
    /// Returns a new flag set with all flags from both `self` and `other`.
    pub const fn insert(self, other: Self) -> Self { Self(self.0 | other.0) }
    /// Returns a new flag set with `other`'s flags cleared from `self`.
    pub const fn remove(self, other: Self) -> Self { Self(self.0 & !other.0) }
}

// ─── DLPack C-ABI types ───────────────────────────────────────────────────────

/// C-ABI compatible representation of a DLDataType (used in DLTensor).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawDLDataType {
    pub code:  u8,
    pub bits:  u8,
    pub lanes: u16,
}

impl From<DLDataType> for RawDLDataType {
    fn from(dt: DLDataType) -> Self {
        let (code, bits, lanes) = dt.to_raw();
        Self { code, bits, lanes }
    }
}

/// C-ABI DLDevice.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawDLDevice {
    pub device_type: i32,
    pub device_id:   i32,
}

/// C-ABI DLTensor (DLPack v0.8 layout).
#[repr(C)]
pub struct DLTensor {
    pub data:        *mut std::ffi::c_void,
    pub device:      RawDLDevice,
    pub ndim:        i32,
    pub dtype:       RawDLDataType,
    pub shape:       *const i64,
    pub strides:     *const i64,
    pub byte_offset: u64,
}

/// Context kept alive for the lifetime of an exported `DLManagedTensor`.
struct DLExportCtx {
    /// Keeps the backing buffer alive until the DLPack consumer is done.
    _raw:    Arc<RawBuffer>,
    shape:   Vec<i64>,
    strides: Vec<i64>,
}

/// C-ABI DLManagedTensor (DLPack v0.8).
#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor:   DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter:     Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// Called by the DLPack consumer when it no longer needs the tensor.
unsafe extern "C" fn dlmanaged_deleter(ptr: *mut DLManagedTensor) {
    if ptr.is_null() { return; }
    unsafe {
        let managed = &*ptr;
        if !managed.manager_ctx.is_null() {
            drop(Box::from_raw(managed.manager_ctx as *mut DLExportCtx));
        }
        drop(Box::from_raw(ptr));
    }
}

// ─── RawBuffer ────────────────────────────────────────────────────────────────

/// The source of a `RawBuffer`'s backing bytes.
#[allow(dead_code)]
enum BufferSource {
    /// Memory owned by mohu's allocator.
    Owned(AllocHandle),
    /// Externally owned memory imported via DLPack.
    /// The deleter is invoked when this `RawBuffer` is dropped.
    DLPack {
        managed: *mut DLManagedTensor,
    },
}

// SAFETY: *mut DLManagedTensor is owned exclusively by this RawBuffer.
// The DLPack contract requires that callers do not alias the pointer.
unsafe impl Send for BufferSource {}
unsafe impl Sync for BufferSource {}

impl Drop for BufferSource {
    fn drop(&mut self) {
        if let BufferSource::DLPack { managed } = self {
            if !(*managed).is_null() {
                let m = unsafe { &**managed };
                if let Some(deleter) = m.deleter {
                    unsafe { deleter(*managed) };
                }
            }
        }
    }
}

/// Raw untyped byte storage — the innermost layer of mohu-buffer.
///
/// Held via `Arc<RawBuffer>` by all `Buffer` instances that share the data.
pub struct RawBuffer {
    source: BufferSource,
    /// Pointer to the usable start of the data (may be offset into the
    /// DLPack allocation).
    ptr:    NonNull<u8>,
    /// Number of usable bytes.
    nbytes: usize,
}

// SAFETY: RawBuffer exclusively owns its bytes through Arc + BufferSource.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

impl RawBuffer {
    /// Creates a new owned `RawBuffer` by allocating `nbytes` bytes.
    fn alloc(nbytes: usize, zeroed: bool) -> MohuResult<Self> {
        let handle = if zeroed {
            AllocHandle::alloc_zeroed(nbytes, SIMD_ALIGN)?
        } else {
            AllocHandle::alloc(nbytes, SIMD_ALIGN)?
        };
        let ptr = if nbytes == 0 {
            NonNull::dangling()
        } else {
            handle.as_non_null()?
        };
        Ok(Self { source: BufferSource::Owned(handle), ptr, nbytes })
    }

    /// Wraps an externally owned DLPack pointer.
    ///
    /// # Safety
    ///
    /// - `managed` must be a valid, non-null `*mut DLManagedTensor`.
    /// - The bytes pointed to by `ptr` must remain valid until the deleter
    ///   in `managed` is invoked.
    unsafe fn from_dlpack_ptr(
        managed: *mut DLManagedTensor,
        ptr:     NonNull<u8>,
        nbytes:  usize,
    ) -> Self {
        Self {
            source: BufferSource::DLPack { managed },
            ptr,
            nbytes,
        }
    }

    /// Returns the data pointer.
    #[inline] pub fn as_ptr(&self) -> *const u8 { self.ptr.as_ptr() }
    /// Returns a mutable data pointer (caller must ensure exclusive access).
    #[inline] pub fn as_mut_ptr(&self) -> *mut u8 { self.ptr.as_ptr() }
    /// Returns the byte capacity of this raw buffer.
    #[inline] pub fn nbytes(&self) -> usize { self.nbytes }

    /// Returns `true` if the backing memory is SIMD-aligned.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % SIMD_ALIGN == 0
    }

    /// Returns `true` if this is an externally-managed (DLPack) buffer.
    pub fn is_external(&self) -> bool {
        matches!(self.source, BufferSource::DLPack { .. })
    }
}

impl std::fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawBuffer")
            .field("ptr",    &format_args!("{:p}", self.ptr))
            .field("nbytes", &self.nbytes)
            .field("external", &self.is_external())
            .finish()
    }
}

// ─── Buffer ───────────────────────────────────────────────────────────────────

/// A reference-counted, typed, N-dimensional array buffer.
///
/// `Buffer` is the primary memory-management type in mohu.  `mohu-array` adds
/// a typed API on top; `Buffer` deals exclusively with bytes, dtypes, shapes,
/// and strides.
///
/// # Clone / share semantics
///
/// `Buffer::clone()` returns a shallow copy (increments the Arc count).
/// To get an independent copy of the data, call [`make_unique`](Buffer::make_unique).
#[derive(Debug)]
pub struct Buffer {
    raw:    Arc<RawBuffer>,
    dtype:  DType,
    layout: Layout,
    flags:  BufferFlags,
}

impl Buffer {
    // ─── Allocation ───────────────────────────────────────────────────────────

    /// Allocates an uninitialized buffer with `dtype` and `shape` in `order` memory.
    pub fn alloc(dtype: DType, shape: &[usize], order: Order) -> MohuResult<Self> {
        let nbytes = contiguous_nbytes(shape, dtype.itemsize())?;
        let raw = Arc::new(RawBuffer::alloc(nbytes, false)?);
        let layout = match order {
            Order::C => Layout::new_c(shape, dtype.itemsize())?,
            Order::F => Layout::new_f(shape, dtype.itemsize())?,
        };
        let flags = Self::compute_flags(&raw, &layout);
        Ok(Self { raw, dtype, layout, flags })
    }

    /// Allocates a zeroed buffer with `dtype` and `shape`.
    pub fn zeros(dtype: DType, shape: &[usize]) -> MohuResult<Self> {
        let nbytes = contiguous_nbytes(shape, dtype.itemsize())?;
        let raw = Arc::new(RawBuffer::alloc(nbytes, true)?);
        let layout = Layout::new_c(shape, dtype.itemsize())?;
        let flags = Self::compute_flags(&raw, &layout);
        Ok(Self { raw, dtype, layout, flags })
    }

    /// Allocates a buffer filled with the one-value for `dtype`.
    ///
    /// Uses Rayon to parallel-fill for large arrays.
    pub fn ones(dtype: DType, shape: &[usize]) -> MohuResult<Self> {
        let mut buf = Self::alloc(dtype, shape, Order::C)?;
        crate::ops::fill_one(&mut buf)?;
        Ok(buf)
    }

    /// Allocates a buffer filled with `fill_bytes` repeated for each element.
    ///
    /// `fill_bytes.len()` must equal `dtype.itemsize()`.
    pub fn full(
        dtype:      DType,
        shape:      &[usize],
        fill_bytes: &[u8],
    ) -> MohuResult<Self> {
        if fill_bytes.len() != dtype.itemsize() {
            return Err(MohuError::bug(format!(
                "Buffer::full: fill_bytes.len()={} != dtype.itemsize()={}",
                fill_bytes.len(), dtype.itemsize()
            )));
        }
        let mut buf = Self::alloc(dtype, shape, Order::C)?;
        crate::ops::fill_raw(&mut buf, fill_bytes)?;
        Ok(buf)
    }

    // ─── Construction from existing data ─────────────────────────────────────

    /// Copies elements from a typed slice into a new C-contiguous buffer.
    pub fn from_slice<T: Scalar>(data: &[T]) -> MohuResult<Self> {
        let dtype  = T::DTYPE;
        let shape  = [data.len()];
        let nbytes = data.len() * dtype.itemsize();
        let raw    = Arc::new(RawBuffer::alloc(nbytes, false)?);
        // SAFETY: raw has exactly nbytes of valid writable memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                raw.as_mut_ptr(),
                nbytes,
            );
        }
        let layout = Layout::new_c(&shape, dtype.itemsize())?;
        let flags  = Self::compute_flags(&raw, &layout)
            .insert(BufferFlags::WRITEABLE);
        Ok(Self { raw, dtype, layout, flags })
    }

    /// Copies a 2D slice-of-slices into a row-major buffer.
    pub fn from_slice_2d<T: Scalar>(data: &[&[T]]) -> MohuResult<Self> {
        if data.is_empty() {
            return Self::zeros(T::DTYPE, &[0, 0]);
        }
        let cols = data[0].len();
        for (_row_idx, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(MohuError::ShapeMismatch {
                    expected: vec![cols],
                    got:      vec![row.len()],
                });
            }
        }
        let rows   = data.len();
        let dtype  = T::DTYPE;
        let shape  = [rows, cols];
        let nbytes = rows * cols * dtype.itemsize();
        let raw    = Arc::new(RawBuffer::alloc(nbytes, false)?);
        unsafe {
            let mut dst = raw.as_mut_ptr();
            for row in data {
                let row_bytes = row.len() * dtype.itemsize();
                std::ptr::copy_nonoverlapping(
                    row.as_ptr() as *const u8,
                    dst,
                    row_bytes,
                );
                dst = dst.add(row_bytes);
            }
        }
        let layout = Layout::new_c(&shape, dtype.itemsize())?;
        let flags  = Self::compute_flags(&raw, &layout)
            .insert(BufferFlags::WRITEABLE);
        Ok(Self { raw, dtype, layout, flags })
    }

    /// Wraps a `Vec<T>` by copying it into a mohu buffer.
    pub fn from_vec<T: Scalar>(data: Vec<T>) -> MohuResult<Self> {
        Self::from_slice(&data)
    }

    /// Wraps raw bytes as a buffer.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for `nbytes` bytes for the lifetime implied by
    ///   the returned `Buffer`.
    /// - `layout` must describe a valid view into those bytes.
    /// - The caller must not free `ptr` while the returned `Buffer` is alive.
    ///
    /// Prefer `from_slice` or DLPack import for safer alternatives.
    pub unsafe fn from_raw_parts(
        ptr:    NonNull<u8>,
        nbytes: usize,
        dtype:  DType,
        layout: Layout,
    ) -> Self {
        // We create a fake AllocHandle that owns nothing — zero-size handle.
        // Caller is responsible for the actual lifetime.
        let raw = Arc::new(RawBuffer {
            source: BufferSource::Owned(
                AllocHandle::alloc(0, SIMD_ALIGN)
                    .expect("zero-size alloc never fails")
            ),
            ptr,
            nbytes,
        });
        let flags = Self::compute_flags(&raw, &layout);
        Self { raw, dtype, layout, flags }
    }

    // ─── DLPack import ────────────────────────────────────────────────────────

    /// Imports a DLPack tensor as a zero-copy `Buffer`.
    ///
    /// The returned `Buffer` holds a reference to the `DLManagedTensor` and
    /// calls its deleter when the last Arc reference is dropped.
    ///
    /// # Safety
    ///
    /// `managed` must be a valid, non-null `*mut DLManagedTensor` whose bytes
    /// remain valid until the deleter is called.
    pub unsafe fn from_dlpack(managed: *mut DLManagedTensor) -> MohuResult<Self> {
        if managed.is_null() {
            return Err(MohuError::DLPackNullPointer);
        }
        let (tensor_device, tensor_dtype, tensor_ndim, tensor_data,
             tensor_byte_offset, tensor_shape, tensor_strides) = unsafe {
            let m = &*managed;
            let t = &m.dl_tensor;
            (t.device, t.dtype, t.ndim, t.data, t.byte_offset, t.shape, t.strides)
        };

        assert_cpu_device(tensor_device.device_type)?;

        let dtype = DType::from_dlpack(
            tensor_dtype.code,
            tensor_dtype.bits,
            tensor_dtype.lanes,
        )?;

        let ndim = tensor_ndim as usize;
        let byte_offset = tensor_byte_offset as usize;
        let base_ptr = unsafe { (tensor_data as *mut u8).add(byte_offset) };
        let ptr = NonNull::new(base_ptr).ok_or_else(|| {
            MohuError::DLPackInvalid("DLTensor.data is null".to_string())
        })?;

        let shape: Vec<usize> = if ndim == 0 {
            vec![]
        } else {
            unsafe { std::slice::from_raw_parts(tensor_shape, ndim) }
                .iter()
                .map(|&d| d as usize)
                .collect()
        };

        // Build layout from DLPack strides (element counts → bytes).
        let layout = if tensor_strides.is_null() {
            Layout::new_c(&shape, dtype.itemsize())?
        } else {
            let raw_strides = unsafe { std::slice::from_raw_parts(tensor_strides, ndim) };
            let byte_strides: Vec<isize> = raw_strides
                .iter()
                .map(|&s| (s * dtype.itemsize() as i64) as isize)
                .collect();
            Layout::new_custom(&shape, &byte_strides, 0, dtype.itemsize())?
        };

        let size = layout.size();
        let nbytes = size * dtype.itemsize();

        let raw = Arc::new(unsafe { RawBuffer::from_dlpack_ptr(managed, ptr, nbytes) });
        let mut flags = Self::compute_flags(&raw, &layout);
        flags = flags.remove(BufferFlags::WRITEABLE);

        Ok(Self { raw, dtype, layout, flags })
    }

    // ─── DLPack export ────────────────────────────────────────────────────────

    /// Exports this buffer as a `DLManagedTensor` for zero-copy handoff to
    /// PyTorch, JAX, CuPy, or any DLPack consumer.
    ///
    /// The returned pointer owns a `DLManagedTensor` on the heap.
    /// The consumer **must** call the `deleter` function when done.
    ///
    /// # DLPack contract
    ///
    /// The caller must ensure the consumer calls `deleter` exactly once.
    /// Dropping the returned pointer without calling the deleter leaks memory.
    pub fn to_dlpack(&self) -> MohuResult<*mut DLManagedTensor> {
        let dl_dtype = RawDLDataType::from(self.dtype.to_dlpack());
        let ndim     = self.layout.ndim();

        let shape: Vec<i64> = self.layout.shape().iter()
            .map(|&d| d as i64).collect();

        // Convert byte strides to element strides.
        let itemsize = self.dtype.itemsize() as isize;
        let strides: Vec<i64> = self.layout.strides().iter()
            .map(|&s| (s / itemsize) as i64).collect();

        let ctx = Box::new(DLExportCtx {
            _raw:    Arc::clone(&self.raw),
            shape,
            strides,
        });
        let ctx_ptr = Box::into_raw(ctx);

        let data_ptr = unsafe {
            self.raw.as_mut_ptr().add(self.layout.offset()) as *mut std::ffi::c_void
        };

        let dl_tensor = DLTensor {
            data:        data_ptr,
            device:      RawDLDevice { device_type: 1, device_id: 0 },
            ndim:        ndim as i32,
            dtype:       dl_dtype,
            shape:       unsafe { (*ctx_ptr).shape.as_ptr() },
            strides:     unsafe { (*ctx_ptr).strides.as_ptr() },
            byte_offset: 0,
        };

        let managed = Box::new(DLManagedTensor {
            dl_tensor,
            manager_ctx: ctx_ptr as *mut std::ffi::c_void,
            deleter:     Some(dlmanaged_deleter),
        });

        Ok(Box::into_raw(managed))
    }

    // ─── Properties ───────────────────────────────────────────────────────────

    /// Returns the element data type of this buffer.
    #[inline] pub fn dtype(&self)    -> DType   { self.dtype }
    /// Returns a reference to this buffer's layout descriptor.
    #[inline] pub fn layout(&self)   -> &Layout { &self.layout }
    /// Returns the shape of this buffer as a slice of dimension sizes.
    #[inline] pub fn shape(&self)    -> &[usize] { self.layout.shape() }
    /// Returns the byte strides of this buffer.
    #[inline] pub fn strides(&self)  -> &[isize] { self.layout.strides() }
    /// Returns the number of dimensions (axes) of this buffer.
    #[inline] pub fn ndim(&self)     -> usize    { self.layout.ndim() }
    /// Returns the total number of elements in this buffer.
    #[inline] pub fn len(&self)      -> usize    { self.layout.size() }
    /// Returns the total byte size of this buffer's data (`len * itemsize`).
    #[inline] pub fn nbytes(&self)   -> usize    { self.layout.nbytes() }
    /// Returns the byte size of a single element.
    #[inline] pub fn itemsize(&self) -> usize    { self.dtype.itemsize() }
    /// Returns the byte offset from the backing buffer start to element `[0, …, 0]`.
    #[inline] pub fn offset(&self)   -> usize    { self.layout.offset() }
    /// Returns the bitfield flags describing this buffer's properties.
    #[inline] pub fn flags(&self)    -> BufferFlags { self.flags }

    /// Returns `true` if any dimension is zero (zero-element buffer).
    pub fn is_empty(&self)         -> bool { self.layout.is_empty() }
    /// Returns `true` if this buffer is writeable.
    pub fn is_writeable(&self)     -> bool { self.flags.contains(BufferFlags::WRITEABLE) }
    /// Returns `true` if this buffer is C-contiguous (row-major).
    pub fn is_c_contiguous(&self)  -> bool { self.layout.is_c_contiguous() }
    /// Returns `true` if this buffer is Fortran-contiguous (column-major).
    pub fn is_f_contiguous(&self)  -> bool { self.layout.is_f_contiguous() }
    /// Returns `true` if this buffer is contiguous in either C or F order.
    pub fn is_contiguous(&self)    -> bool { self.layout.is_contiguous() }
    /// Returns `true` if the backing memory is SIMD-aligned.
    pub fn is_aligned(&self)       -> bool { self.flags.contains(BufferFlags::ALIGNED) }
    /// Returns `true` if the backing memory is shared with other `Buffer` instances.
    pub fn is_shared(&self)        -> bool { Arc::strong_count(&self.raw) > 1 }

    /// Returns a raw const pointer to element `[0, 0, …, 0]`.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.raw.as_ptr().add(self.layout.offset()) }
    }

    /// Returns a raw mutable pointer to element `[0, 0, …, 0]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other live references into this buffer exist.
    #[inline]
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        unsafe { self.raw.as_mut_ptr().add(self.layout.offset()) }
    }

    // ─── Typed element access ─────────────────────────────────────────────────

    /// Interprets the buffer's data as a slice of `T`.
    ///
    /// Returns `Err(DTypeMismatch)` if `T::DTYPE != self.dtype`.
    /// Returns `Err(NonContiguous)` if the buffer is not C-contiguous.
    pub fn as_slice<T: Scalar>(&self) -> MohuResult<&[T]> {
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        if !self.is_c_contiguous() {
            return Err(MohuError::NonContiguous);
        }
        let ptr = self.as_ptr() as *const T;
        // SAFETY: layout + dtype checks above guarantee validity.
        Ok(unsafe { std::slice::from_raw_parts(ptr, self.len()) })
    }

    /// Interprets the buffer's data as a mutable slice of `T`.
    ///
    /// Returns `Err(ReadOnly)` if the buffer is not writeable.
    pub fn as_mut_slice<T: Scalar>(&mut self) -> MohuResult<&mut [T]> {
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        if !self.is_writeable() {
            return Err(MohuError::ReadOnly);
        }
        if !self.is_c_contiguous() {
            return Err(MohuError::NonContiguous);
        }
        // Ensure we have unique access before handing out a mutable slice.
        self.make_unique()?;
        let ptr = unsafe { self.as_mut_ptr() } as *mut T;
        let len = self.len();
        // SAFETY: uniqueness ensured above, layout + dtype checks done.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    /// Returns the element at `indices` as type `T`.
    ///
    /// Works on non-contiguous arrays (uses stride-based offset).
    pub fn get<T: Scalar>(&self, indices: &[usize]) -> MohuResult<T> {
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        let off = self.layout.byte_offset(indices)?;
        let ptr = unsafe { self.raw.as_ptr().add(off) as *const T };
        // SAFETY: offset is bounds-checked above.
        Ok(unsafe { ptr.read_unaligned() })
    }

    /// Sets the element at `indices` to `value`.
    pub fn set<T: Scalar>(&mut self, indices: &[usize], value: T) -> MohuResult<()> {
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        if !self.is_writeable() {
            return Err(MohuError::ReadOnly);
        }
        self.make_unique()?;
        let off = self.layout.byte_offset(indices)?;
        let ptr = unsafe { self.raw.as_mut_ptr().add(off) as *mut T };
        // SAFETY: offset is bounds-checked, make_unique guarantees exclusivity.
        unsafe { ptr.write_unaligned(value) };
        Ok(())
    }

    // ─── View operations (zero-copy) ──────────────────────────────────────────

    /// Returns a shallow clone of this buffer — increments the Arc count.
    ///
    /// Both the original and the clone share the same backing bytes.
    /// To get an independent copy, call [`make_unique`](Self::make_unique).
    pub fn share(&self) -> Self {
        Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout: self.layout.clone(),
            flags:  self.flags.remove(BufferFlags::WRITEABLE), // shared = read-only
        }
    }

    /// Makes this buffer uniquely owned, copying the data if necessary.
    ///
    /// After this call, `is_shared()` is `false` and the buffer is writeable.
    pub fn make_unique(&mut self) -> MohuResult<()> {
        if Arc::strong_count(&self.raw) == 1 {
            self.flags = self.flags.insert(BufferFlags::WRITEABLE);
            return Ok(());
        }
        // Copy-on-write: allocate a new contiguous buffer and copy.
        let new_buf = self.to_contiguous()?;
        *self = new_buf;
        Ok(())
    }

    /// Copies this buffer (including non-contiguous data) into a new
    /// C-contiguous owned buffer.
    pub fn to_contiguous(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.layout.shape(), Order::C)?;
        crate::ops::copy_to_contiguous(self, &mut dst)?;
        Ok(dst)
    }

    /// Sets the writeable flag.  Errors if the buffer is shared (Arc count > 1).
    pub fn set_writeable(&mut self, writeable: bool) -> MohuResult<()> {
        if writeable && self.is_shared() {
            return Err(MohuError::CannotResizeShared);
        }
        if writeable {
            self.flags = self.flags.insert(BufferFlags::WRITEABLE);
        } else {
            self.flags = self.flags.remove(BufferFlags::WRITEABLE);
        }
        Ok(())
    }

    // ─── Layout transformations (zero-copy) ──────────────────────────────────

    /// Returns a transposed view (reverses axis order, no copy).
    pub fn transpose(&self) -> Self {
        Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout: self.layout.transpose(),
            flags:  self.flags.remove(BufferFlags::WRITEABLE)
                               .remove(BufferFlags::C_CONTIGUOUS),
        }
    }

    /// Returns a view with axes permuted according to `axes`.
    pub fn permute(&self, axes: &[usize]) -> MohuResult<Self> {
        let layout = self.layout.permute(axes)?;
        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags:  self.flags.remove(BufferFlags::WRITEABLE)
                               .remove(BufferFlags::C_CONTIGUOUS),
        })
    }

    /// Returns a reshaped view.  Requires C-contiguous layout.
    pub fn reshape(&self, new_shape: &[usize]) -> MohuResult<Self> {
        let layout = self.layout.reshape(new_shape)?;
        let flags  = Self::compute_flags(&self.raw, &layout);
        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags,
        })
    }

    /// Returns a slice along `axis` with the given `SliceArg`.
    pub fn slice_axis(&self, axis: usize, arg: SliceArg) -> MohuResult<Self> {
        let layout = self.layout.slice_axis(axis, arg)?;
        let flags  = self.flags
            .remove(BufferFlags::WRITEABLE)
            .remove(BufferFlags::C_CONTIGUOUS)
            .remove(BufferFlags::F_CONTIGUOUS);
        Ok(Self { raw: Arc::clone(&self.raw), dtype: self.dtype, layout, flags })
    }

    /// Returns a broadcast view to `new_shape`.  Broadcast axes are read-only.
    pub fn broadcast_to(&self, new_shape: &[usize]) -> MohuResult<Self> {
        let layout = self.layout.broadcast_to(new_shape)?;
        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags:  self.flags.remove(BufferFlags::WRITEABLE),
        })
    }

    /// Inserts a new axis of size 1 at position `axis`.
    pub fn expand_dims(&self, axis: usize) -> MohuResult<Self> {
        let layout = self.layout.expand_dims(axis)?;
        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags:  self.flags,
        })
    }

    /// Removes all axes of size 1.
    pub fn squeeze(&self) -> Self {
        Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout: self.layout.squeeze(),
            flags:  self.flags,
        }
    }

    // ─── Type cast ────────────────────────────────────────────────────────────

    /// Returns a new buffer with elements cast to `target`.
    ///
    /// If `target == self.dtype`, returns a contiguous copy.
    pub fn cast(&self, target: DType, mode: CastMode) -> MohuResult<Self> {
        if target == self.dtype {
            return self.to_contiguous();
        }
        let mut dst = Self::alloc(target, self.layout.shape(), Order::C)?;
        crate::ops::cast_copy(self, &mut dst, mode)?;
        Ok(dst)
    }

    // ─── Bulk data access ─────────────────────────────────────────────────────

    /// Copies all elements into a `Vec<T>`, performing stride traversal if
    /// the array is non-contiguous.
    pub fn to_vec<T: Scalar>(&self) -> MohuResult<Vec<T>> {
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        if self.is_c_contiguous() {
            let slice = self.as_slice::<T>()?;
            return Ok(slice.to_vec());
        }
        // Non-contiguous: iterate byte offsets.
        let mut out = Vec::with_capacity(self.len());
        for off in crate::strides::StridedByteIter::new(
            self.layout.shape(),
            self.layout.strides(),
            self.layout.offset(),
        ) {
            let ptr = unsafe { self.raw.as_ptr().add(off) as *const T };
            out.push(unsafe { ptr.read_unaligned() });
        }
        Ok(out)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────────

    fn compute_flags(raw: &RawBuffer, layout: &Layout) -> BufferFlags {
        let mut f = BufferFlags::empty()
            .insert(BufferFlags::WRITEABLE)
            .insert(BufferFlags::OWNS_DATA);
        if raw.is_aligned() { f = f.insert(BufferFlags::ALIGNED); }
        if layout.is_c_contiguous() { f = f.insert(BufferFlags::C_CONTIGUOUS); }
        if layout.is_f_contiguous() { f = f.insert(BufferFlags::F_CONTIGUOUS); }
        f
    }
}

impl Clone for Buffer {
    /// Shallow clone — increments the Arc count.  Call [`make_unique`](Buffer::make_unique)
    /// for a deep (independent) copy.
    fn clone(&self) -> Self {
        Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout: self.layout.clone(),
            flags:  self.flags.remove(BufferFlags::WRITEABLE),
        }
    }
}

// SAFETY: Buffer holds its data via Arc<RawBuffer> which is Send+Sync.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

// ─── Compile-time checks ──────────────────────────────────────────────────────

const _ASSERT_SEND_SYNC: () = {
    const fn check<T: Send + Sync>() {}
    check::<Buffer>();
};

// ─── Extended constructors ────────────────────────────────────────────────────

impl Buffer {
    /// Creates a 1-D buffer `[start, start+step, …]` stopping before `stop`.
    ///
    /// Equivalent to `np.arange`.  The number of elements is
    /// `ceil((stop - start) / step)`.
    ///
    /// ```rust,ignore
    /// let a = Buffer::arange(0.0, 10.0, 2.0, DType::F64)?; // [0, 2, 4, 6, 8]
    /// ```
    pub fn arange(start: f64, stop: f64, step: f64, dtype: DType) -> MohuResult<Self> {
        if step == 0.0 {
            return Err(MohuError::bug("arange: step must be non-zero"));
        }
        let n = if step > 0.0 {
            ((stop - start) / step).ceil().max(0.0) as usize
        } else {
            ((start - stop) / (-step)).ceil().max(0.0) as usize
        };

        let buf = Self::alloc(DType::F64, &[n], Order::C)?;
        if n > 0 {
            let ptr   = unsafe { buf.as_mut_ptr() as *mut f64 };
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
            // Index-based so every element is independent — safe for Rayon.
            use rayon::prelude::*;
            slice.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = start + i as f64 * step;
            });
        }

        if dtype == DType::F64 { Ok(buf) } else { buf.cast(dtype, CastMode::Unsafe) }
    }

    /// Creates a 1-D buffer of `n` evenly-spaced values from `start` to `stop`.
    ///
    /// Equivalent to `np.linspace`.  If `endpoint` is `true`, `stop` is included.
    pub fn linspace(
        start:    f64,
        stop:     f64,
        n:        usize,
        endpoint: bool,
        dtype:    DType,
    ) -> MohuResult<Self> {
        if n == 0 {
            return Self::alloc(dtype, &[0], Order::C);
        }
        let div  = if endpoint && n > 1 { (n - 1) as f64 } else { n as f64 };
        let span = stop - start;

        let buf = Self::alloc(DType::F64, &[n], Order::C)?;
        {
            let ptr   = unsafe { buf.as_mut_ptr() as *mut f64 };
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
            use rayon::prelude::*;
            slice.par_iter_mut().enumerate().for_each(|(i, v)| {
                *v = start + i as f64 * span / div;
            });
            if endpoint && n > 1 {
                slice[n - 1] = stop; // pin the exact endpoint
            }
        }

        if dtype == DType::F64 { Ok(buf) } else { buf.cast(dtype, CastMode::Unsafe) }
    }

    /// Creates an `n × m` identity matrix with ones on diagonal `k`.
    ///
    /// `k = 0` → main diagonal, `k > 0` → super-diagonal, `k < 0` → sub-diagonal.
    /// Equivalent to `np.eye`.
    pub fn eye(n: usize, m: usize, k: i64, dtype: DType) -> MohuResult<Self> {
        let buf = Self::zeros(dtype, &[n, m])?;
        if n == 0 || m == 0 { return Ok(buf); }

        let one_bytes = dtype_one_bytes(dtype);
        let itemsize  = dtype.itemsize();
        let raw_ptr   = unsafe { buf.as_mut_ptr() };

        let (row_start, col_start) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };
        let diag_len = (n - row_start).min(m.saturating_sub(col_start));

        for i in 0..diag_len {
            let off = buf.layout.byte_offset(&[row_start + i, col_start + i])?;
            unsafe {
                std::ptr::copy_nonoverlapping(one_bytes.as_ptr(), raw_ptr.add(off), itemsize);
            }
        }
        Ok(buf)
    }

    /// Constructs a 2-D diagonal matrix from a 1-D buffer `v`, with offset `k`.
    ///
    /// Equivalent to `np.diag(v, k)` when `v` is 1-D.
    pub fn diag(v: &Buffer, k: i64) -> MohuResult<Self> {
        if v.ndim() != 1 {
            return Err(MohuError::bug("Buffer::diag: input must be 1-D"));
        }
        let n    = v.len();
        let size = if k >= 0 { n + k as usize } else { n + (-k) as usize };
        let out = Self::zeros(v.dtype(), &[size, size])?;

        let itemsize = v.dtype().itemsize();
        let src_raw  = v.as_ptr();
        let dst_raw  = unsafe { out.as_mut_ptr() };

        let (row_start, col_start) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };

        for i in 0..n {
            let src_off = i * itemsize;
            let dst_off = out.layout.byte_offset(&[row_start + i, col_start + i])?;
            unsafe {
                std::ptr::copy_nonoverlapping(src_raw.add(src_off), dst_raw.add(dst_off), itemsize);
            }
        }
        Ok(out)
    }

    /// Returns a zero-copy view of the diagonal with offset `k` as a 1-D buffer.
    ///
    /// Stride of the result = `row_stride + col_stride` of the input.
    /// Works for any 2-D buffer (contiguous or not).
    pub fn diagonal(&self, k: i64) -> MohuResult<Self> {
        if self.ndim() < 2 {
            return Err(MohuError::bug("diagonal: requires at least 2 dimensions"));
        }
        let nd  = self.ndim();
        let n   = self.shape()[nd - 2];
        let m   = self.shape()[nd - 1];

        let (row_start, col_start) = if k >= 0 {
            (0usize, k as usize)
        } else {
            ((-k) as usize, 0usize)
        };
        let diag_len = (n.saturating_sub(row_start)).min(m.saturating_sub(col_start));

        // Stride along the diagonal = last-two-axes strides summed.
        let rs = self.strides()[nd - 2];
        let cs = self.strides()[nd - 1];
        let diag_stride = rs + cs;

        // Starting byte offset.
        let mut start_idx = vec![0usize; nd];
        start_idx[nd - 2] = row_start;
        start_idx[nd - 1] = col_start;
        let start_off = self.layout.byte_offset(&start_idx)?;

        // Build new shape/strides: batch dims + [diag_len].
        let mut new_shape: Vec<usize> = self.shape()[..nd - 2].to_vec();
        new_shape.push(diag_len);
        let mut new_strides: Vec<isize> = self.strides()[..nd - 2].to_vec();
        new_strides.push(diag_stride);

        let layout = Layout::new_custom(
            &new_shape,
            &new_strides,
            start_off,
            self.layout.itemsize(),
        )?;

        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags:  self.flags
                .remove(BufferFlags::WRITEABLE)
                .remove(BufferFlags::C_CONTIGUOUS)
                .remove(BufferFlags::F_CONTIGUOUS),
        })
    }

    /// Returns a zero-copy view with axis `axis` reversed (negative stride).
    ///
    /// No data is copied.  The returned buffer is non-writeable (it is a view).
    /// Equivalent to `a[::-1]` in NumPy.
    pub fn flip(&self, axis: usize) -> MohuResult<Self> {
        if axis >= self.ndim() {
            return Err(MohuError::bug(format!(
                "flip: axis {axis} out of bounds for ndim {}",
                self.ndim()
            )));
        }
        let dim   = self.shape()[axis];
        let mut new_strides: Vec<isize> = self.strides().to_vec();
        // New offset = old offset + (dim-1) * old_stride[axis]
        let offset_delta = if dim > 0 {
            (dim as isize - 1) * new_strides[axis]
        } else {
            0
        };
        let new_offset = self.layout.offset() as isize + offset_delta;
        new_strides[axis] = -new_strides[axis];

        if new_offset < 0 {
            return Err(MohuError::bug("flip: computed offset is negative"));
        }

        let layout = Layout::new_custom(
            self.shape(),
            &new_strides,
            new_offset as usize,
            self.layout.itemsize(),
        )?;

        Ok(Self {
            raw:    Arc::clone(&self.raw),
            dtype:  self.dtype,
            layout,
            flags:  self.flags
                .remove(BufferFlags::WRITEABLE)
                .remove(BufferFlags::C_CONTIGUOUS)
                .remove(BufferFlags::F_CONTIGUOUS),
        })
    }

    /// Creates a lower-triangular copy of this 2-D buffer.
    ///
    /// Elements above diagonal `k` are zeroed.
    /// `k = 0` keeps the main diagonal; `k < 0` zeros more; `k > 0` keeps more.
    pub fn tril(&self, k: i64) -> MohuResult<Self> {
        if self.ndim() != 2 {
            return Err(MohuError::bug("tril: requires exactly 2 dimensions"));
        }
        let out = self.to_contiguous()?;
        let (rows, cols) = (out.shape()[0], out.shape()[1]);
        let itemsize = out.dtype().itemsize();
        let zero_bytes = vec![0u8; itemsize];
        let ptr = unsafe { out.as_mut_ptr() };
        for r in 0..rows {
            for c in 0..cols {
                // zero element if c > r + k
                if c as i64 > r as i64 + k {
                    let off = out.layout.byte_offset(&[r, c])?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(zero_bytes.as_ptr(), ptr.add(off), itemsize);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Creates an upper-triangular copy of this 2-D buffer.
    ///
    /// Elements below diagonal `k` are zeroed.
    pub fn triu(&self, k: i64) -> MohuResult<Self> {
        if self.ndim() != 2 {
            return Err(MohuError::bug("triu: requires exactly 2 dimensions"));
        }
        let out = self.to_contiguous()?;
        let (rows, cols) = (out.shape()[0], out.shape()[1]);
        let itemsize = out.dtype().itemsize();
        let zero_bytes = vec![0u8; itemsize];
        let ptr = unsafe { out.as_mut_ptr() };
        for r in 0..rows {
            for c in 0..cols {
                if (c as i64) < (r as i64 + k) {
                    let off = out.layout.byte_offset(&[r, c])?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(zero_bytes.as_ptr(), ptr.add(off), itemsize);
                    }
                }
            }
        }
        Ok(out)
    }

    // ─── Mutation helpers ─────────────────────────────────────────────────────

    /// Copies elements from `src` into this buffer in-place.
    ///
    /// Shapes must match.  `src` may be non-contiguous.
    pub fn copy_from(&mut self, src: &Buffer) -> MohuResult<()> {
        crate::ops::copy_to_contiguous(src, self)
    }

    /// Fills the main diagonal of a 2-D buffer with `value`.
    ///
    /// Works for non-square matrices.  The buffer must be writeable.
    pub fn fill_diagonal<T: Scalar>(&mut self, value: T) -> MohuResult<()> {
        if self.ndim() != 2 {
            return Err(MohuError::bug("fill_diagonal: requires exactly 2 dimensions"));
        }
        if T::DTYPE != self.dtype {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      self.dtype.to_string(),
            });
        }
        if !self.is_writeable() { return Err(MohuError::ReadOnly); }
        self.make_unique()?;

        let n = self.shape()[0].min(self.shape()[1]);
        for i in 0..n {
            self.set::<T>(&[i, i], value)?;
        }
        Ok(())
    }

    // ─── Reductions ───────────────────────────────────────────────────────────

    /// Sums all elements, returning the result as f64.
    ///
    /// Uses parallel reduction via Rayon.  Non-contiguous arrays are first
    /// converted to contiguous (one allocation).
    pub fn sum_all_f64(&self) -> MohuResult<f64> {
        crate::ops::sum_all_f64(self)
    }

    /// Computes the arithmetic mean of all elements as f64.
    pub fn mean_all_f64(&self) -> MohuResult<f64> {
        let n = self.len();
        if n == 0 { return Ok(f64::NAN); }
        Ok(crate::ops::sum_all_f64(self)? / n as f64)
    }

    /// Returns the minimum element as f64.
    pub fn min_all_f64(&self) -> MohuResult<f64> {
        crate::ops::min_all_f64(self)
    }

    /// Returns the maximum element as f64.
    pub fn max_all_f64(&self) -> MohuResult<f64> {
        crate::ops::max_all_f64(self)
    }

    /// Computes the variance with `ddof` degrees of freedom, as f64.
    ///
    /// `ddof = 0` → population variance, `ddof = 1` → sample variance.
    pub fn var_all_f64(&self, ddof: usize) -> MohuResult<f64> {
        use mohu_dtype::DType;
        if matches!(self.dtype, DType::C64 | DType::C128) {
            return Err(MohuError::UnsupportedDType {
                op: "var_all_f64",
                dtype: self.dtype.to_string(),
            });
        }
        let n = self.len();
        if n <= ddof { return Ok(f64::NAN); }
        let mean = self.mean_all_f64()?;

        // Second pass: sum of squared deviations.
        // Bool: reinterpret as u8 (true=1, false=0) via special pre-check.
        if self.dtype == DType::Bool {
            let c;
            let s: &[u8] = if self.is_c_contiguous() {
                unsafe { std::slice::from_raw_parts(self.as_ptr(), n) }
            } else {
                c = self.to_contiguous()?;
                unsafe { std::slice::from_raw_parts(c.as_ptr(), n) }
            };
            use rayon::prelude::*;
            let ss: f64 = s.par_iter().map(|&x| {
                let d = x as f64 - mean;
                d * d
            }).sum();
            return Ok(ss / (n - ddof) as f64);
        }
        macro_rules! do_var {
            ($T:ty) => {{
                let c;
                let s: &[$T] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const $T, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, n) }
                };
                use rayon::prelude::*;
                let ss: f64 = s.par_iter().map(|&x| {
                    let d = num_traits::cast::<$T, f64>(x).unwrap_or(0.0) - mean;
                    d * d
                }).sum();
                Ok(ss / (n - ddof) as f64)
            }};
        }
        match self.dtype {
            DType::I8   => do_var!(i8),
            DType::I16  => do_var!(i16),
            DType::I32  => do_var!(i32),
            DType::I64  => do_var!(i64),
            DType::U8   => do_var!(u8),
            DType::U16  => do_var!(u16),
            DType::U32  => do_var!(u32),
            DType::U64  => do_var!(u64),
            DType::F16  => do_var!(::half::f16),
            DType::BF16 => do_var!(::half::bf16),
            DType::F32  => do_var!(f32),
            DType::F64  => do_var!(f64),
            _ => unreachable!(),
        }
    }

    /// Computes the standard deviation with `ddof` degrees of freedom.
    pub fn std_all_f64(&self, ddof: usize) -> MohuResult<f64> {
        Ok(self.var_all_f64(ddof)?.sqrt())
    }

    /// Returns the flat index of the minimum element.
    pub fn argmin_flat(&self) -> MohuResult<usize> {
        crate::ops::argmin_flat(self)
    }

    /// Returns the flat index of the maximum element.
    pub fn argmax_flat(&self) -> MohuResult<usize> {
        crate::ops::argmax_flat(self)
    }

    /// Sums over `axis`, returning a buffer with that axis collapsed.
    ///
    /// If `keepdims` is true, the result has a size-1 axis in place of `axis`.
    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> MohuResult<Self> {
        if axis >= self.ndim() {
            return Err(MohuError::bug(format!(
                "sum_axis: axis {axis} out of bounds for ndim {}", self.ndim()
            )));
        }
        // Build output shape
        let mut out_shape: Vec<usize> = self.shape().to_vec();
        let axis_size = out_shape[axis];
        if keepdims { out_shape[axis] = 1; } else { out_shape.remove(axis); }

        let out = Self::zeros(DType::F64, &out_shape)?;
        let out_raw = unsafe { out.as_mut_ptr() as *mut f64 };
        let _ = out.len(); // shape check only; iteration is index-driven

        // For each output element, sum the corresponding slice along axis.
        // Iterate over all positions in the output.
        use crate::strides::NdIndexIter;
        let out_shape_full: Vec<usize> = out.shape().to_vec();

        // Build src index from out index by inserting the axis.
        let itemsize = self.dtype.itemsize();
        let src_raw  = self.as_ptr();

        for (out_flat, out_idx) in NdIndexIter::new(&out_shape_full).enumerate() {
            let mut acc = 0.0f64;
            for k in 0..axis_size {
                // Build source index
                let mut src_idx = out_idx.to_vec();
                if keepdims {
                    src_idx[axis] = k;
                } else {
                    src_idx.insert(axis, k);
                }
                let off = self.layout.byte_offset(&src_idx)?;
                // Read element as f64 via dispatch
                let val = read_as_f64(unsafe { src_raw.add(off) }, self.dtype, itemsize);
                acc += val;
            }
            unsafe { out_raw.add(out_flat).write(acc); }
        }

        Ok(out)
    }

    // ─── Boolean reductions ───────────────────────────────────────────────────

    /// Returns `true` if any element is non-zero (truthy).
    pub fn any(&self) -> MohuResult<bool> {
        use mohu_dtype::DType;
        use rayon::prelude::*;
        let n = self.len();
        // Bool and complex: handle without NumCast.
        match self.dtype {
            DType::Bool => {
                let c;
                let s: &[u8] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr(), n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr(), n) }
                };
                return Ok(s.par_iter().any(|&x| x != 0));
            }
            DType::C64 => {
                let c;
                let s: &[num_complex::Complex<f32>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().any(|x| x.re != 0.0 || x.im != 0.0));
            }
            DType::C128 => {
                let c;
                let s: &[num_complex::Complex<f64>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().any(|x| x.re != 0.0 || x.im != 0.0));
            }
            _ => {}
        }
        macro_rules! do_any {
            ($T:ty) => {{
                let c;
                let s: &[$T] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const $T, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, n) }
                };
                Ok(s.par_iter().any(|&x| {
                    num_traits::cast::<$T, f64>(x).map(|v| v != 0.0).unwrap_or(false)
                }))
            }};
        }
        match self.dtype {
            DType::I8   => do_any!(i8),
            DType::I16  => do_any!(i16),
            DType::I32  => do_any!(i32),
            DType::I64  => do_any!(i64),
            DType::U8   => do_any!(u8),
            DType::U16  => do_any!(u16),
            DType::U32  => do_any!(u32),
            DType::U64  => do_any!(u64),
            DType::F16  => do_any!(::half::f16),
            DType::BF16 => do_any!(::half::bf16),
            DType::F32  => do_any!(f32),
            DType::F64  => do_any!(f64),
            _ => unreachable!(),
        }
    }

    /// Returns `true` if all elements are non-zero (truthy).
    pub fn all(&self) -> MohuResult<bool> {
        use mohu_dtype::DType;
        use rayon::prelude::*;
        let n = self.len();
        match self.dtype {
            DType::Bool => {
                let c;
                let s: &[u8] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr(), n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr(), n) }
                };
                return Ok(s.par_iter().all(|&x| x != 0));
            }
            DType::C64 => {
                let c;
                let s: &[num_complex::Complex<f32>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().all(|x| x.re != 0.0 || x.im != 0.0));
            }
            DType::C128 => {
                let c;
                let s: &[num_complex::Complex<f64>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().all(|x| x.re != 0.0 || x.im != 0.0));
            }
            _ => {}
        }
        macro_rules! do_all {
            ($T:ty) => {{
                let c;
                let s: &[$T] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const $T, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, n) }
                };
                Ok(s.par_iter().all(|&x| {
                    num_traits::cast::<$T, f64>(x).map(|v| v != 0.0).unwrap_or(false)
                }))
            }};
        }
        match self.dtype {
            DType::I8   => do_all!(i8),
            DType::I16  => do_all!(i16),
            DType::I32  => do_all!(i32),
            DType::I64  => do_all!(i64),
            DType::U8   => do_all!(u8),
            DType::U16  => do_all!(u16),
            DType::U32  => do_all!(u32),
            DType::U64  => do_all!(u64),
            DType::F16  => do_all!(::half::f16),
            DType::BF16 => do_all!(::half::bf16),
            DType::F32  => do_all!(f32),
            DType::F64  => do_all!(f64),
            _ => unreachable!(),
        }
    }

    /// Returns the count of non-zero elements.
    pub fn count_nonzero(&self) -> MohuResult<usize> {
        use mohu_dtype::DType;
        use rayon::prelude::*;
        let n = self.len();
        match self.dtype {
            DType::Bool => {
                let c;
                let s: &[u8] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr(), n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr(), n) }
                };
                return Ok(s.par_iter().filter(|&&x| x != 0).count());
            }
            DType::C64 => {
                let c;
                let s: &[num_complex::Complex<f32>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().filter(|x| x.re != 0.0 || x.im != 0.0).count());
            }
            DType::C128 => {
                let c;
                let s: &[num_complex::Complex<f64>] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const _, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const _, n) }
                };
                return Ok(s.par_iter().filter(|x| x.re != 0.0 || x.im != 0.0).count());
            }
            _ => {}
        }
        macro_rules! do_cnz {
            ($T:ty) => {{
                let c;
                let s: &[$T] = if self.is_c_contiguous() {
                    unsafe { std::slice::from_raw_parts(self.as_ptr() as *const $T, n) }
                } else {
                    c = self.to_contiguous()?;
                    unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, n) }
                };
                Ok(s.par_iter().filter(|&&x| {
                    num_traits::cast::<$T, f64>(x).map(|v| v != 0.0).unwrap_or(false)
                }).count())
            }};
        }
        match self.dtype {
            DType::I8   => do_cnz!(i8),
            DType::I16  => do_cnz!(i16),
            DType::I32  => do_cnz!(i32),
            DType::I64  => do_cnz!(i64),
            DType::U8   => do_cnz!(u8),
            DType::U16  => do_cnz!(u16),
            DType::U32  => do_cnz!(u32),
            DType::U64  => do_cnz!(u64),
            DType::F16  => do_cnz!(::half::f16),
            DType::BF16 => do_cnz!(::half::bf16),
            DType::F32  => do_cnz!(f32),
            DType::F64  => do_cnz!(f64),
            _ => unreachable!(),
        }
    }

    // ─── Approximate equality ─────────────────────────────────────────────────

    /// Returns `true` if all elements satisfy `|a - b| <= atol + rtol * |b|`.
    ///
    /// NaN == NaN for this comparison (unlike IEEE 754).
    pub fn allclose(&self, other: &Buffer, rtol: f64, atol: f64) -> MohuResult<bool> {
        if self.shape() != other.shape() {
            return Ok(false);
        }
        let a = self.cast(DType::F64, CastMode::Unsafe)?;
        let b = other.cast(DType::F64, CastMode::Unsafe)?;
        let as_ = a.as_slice::<f64>()?;
        let bs  = b.as_slice::<f64>()?;
        use rayon::prelude::*;
        Ok(as_.par_iter().zip(bs.par_iter()).all(|(a, b)| {
            if a.is_nan() && b.is_nan() { return true; }
            (a - b).abs() <= atol + rtol * b.abs()
        }))
    }

    // ─── Element-wise unary ops ───────────────────────────────────────────────

    /// Returns a new buffer with absolute values.
    pub fn abs(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::abs_copy(self, &mut dst)?;
        Ok(dst)
    }

    /// Returns a new buffer with negated values.
    pub fn neg(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::neg_copy(self, &mut dst)?;
        Ok(dst)
    }

    /// Returns a new buffer with element-wise square root (F32/F64 only).
    pub fn sqrt(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::sqrt_copy(self, &mut dst)?;
        Ok(dst)
    }

    /// Returns a new buffer with element-wise natural log (F32/F64 only).
    pub fn ln(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::ln_copy(self, &mut dst)?;
        Ok(dst)
    }

    /// Returns a new buffer with element-wise exp (F32/F64 only).
    pub fn exp(&self) -> MohuResult<Self> {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::exp_copy(self, &mut dst)?;
        Ok(dst)
    }

    // ─── Arithmetic with scalar ───────────────────────────────────────────────

    /// Returns `self + scalar` (element-wise).
    pub fn add_scalar<T>(&self, scalar: T) -> MohuResult<Self>
    where
        T: Scalar + Copy + Send + Sync + std::ops::Add<Output = T>,
    {
        let mut dst = self.to_contiguous()?;
        crate::ops::add_scalar_inplace::<T>(&mut dst, scalar)?;
        Ok(dst)
    }

    /// Returns `self * scalar` (element-wise).
    pub fn mul_scalar<T>(&self, scalar: T) -> MohuResult<Self>
    where
        T: Scalar + Copy + Send + Sync + std::ops::Mul<Output = T>,
    {
        let mut dst = self.to_contiguous()?;
        crate::ops::mul_scalar_inplace::<T>(&mut dst, scalar)?;
        Ok(dst)
    }

    /// Returns a clipped copy: `clip(self, lo, hi)`.
    pub fn clip_val<T>(&self, lo: T, hi: T) -> MohuResult<Self>
    where
        T: Scalar + PartialOrd + Copy + Send + Sync,
    {
        let mut dst = Self::alloc(self.dtype, self.shape(), Order::C)?;
        crate::ops::clip::<T>(self, &mut dst, lo, hi)?;
        Ok(dst)
    }

    // ─── Memory & layout hints ────────────────────────────────────────────────

    /// Advises the kernel on the access pattern for this buffer's backing memory.
    ///
    /// Delegates to [`AllocHandle::advise`].
    /// No-op for shared buffers (Arc count > 1) or non-Unix platforms.
    pub fn advise(&self, advice: crate::alloc::MmapAdvice) {
        // SAFETY: we don't modify any bytes, just issue a kernel hint.
        // Call madvise directly on the buffer's pointer since AllocHandle is
        // behind an Arc and the inner source field is private.
        #[cfg(unix)]
        unsafe {
            let _ = libc::madvise(
                self.raw.as_mut_ptr().add(self.layout.offset()) as *mut libc::c_void,
                self.nbytes(),
                advice.to_libc(),
            );
        }
        #[cfg(not(unix))]
        let _ = advice;
    }

    /// Software-prefetches the first cache line of this buffer.
    ///
    /// Non-blocking; safe to call on read-only or shared buffers.
    #[inline]
    pub fn prefetch(&self) {
        let ptr = self.as_ptr();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        let _ = ptr;
    }

    // ─── Diagnostics ─────────────────────────────────────────────────────────

    /// Returns a human-readable multi-line description of this buffer,
    /// including shape, dtype, strides, flags, and the first few values.
    pub fn describe(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "Buffer {{");
        let _ = writeln!(s, "  dtype:   {:?}", self.dtype);
        let _ = writeln!(s, "  shape:   {:?}", self.shape());
        let _ = writeln!(s, "  strides: {:?}", self.strides());
        let _ = writeln!(s, "  offset:  {}", self.offset());
        let _ = writeln!(s, "  nbytes:  {}", self.nbytes());
        let _ = writeln!(s, "  flags:   {:08b}", self.flags.0);
        let _ = writeln!(s, "  shared:  {}", self.is_shared());
        let _ = writeln!(s, "  aligned: {}", self.is_aligned());
        let cap = 8.min(self.len());
        if cap > 0 {
            let _ = write!(s, "  data[:{}]: [", cap);
            for i in 0..cap {
                if i > 0 { let _ = write!(s, ", "); }
                // Convert flat index to multi-dim and read as f64 for display
                let idx = flat_to_indices(i, self.shape());
                if let Ok(off) = self.layout.byte_offset(&idx) {
                    let v = read_as_f64(unsafe { self.raw.as_ptr().add(off) }, self.dtype, self.dtype.itemsize());
                    let _ = write!(s, "{v:.4}");
                }
            }
            let _ = writeln!(s, "]");
        }
        let _ = writeln!(s, "}}");
        s
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl std::fmt::Display for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "array(")?;
        fmt_buffer_data(self, f, self.shape(), &mut vec![0usize; self.ndim()], 0, 0)?;
        write!(f, ", dtype={:?})", self.dtype)
    }
}

fn fmt_buffer_data(
    buf:     &Buffer,
    f:       &mut std::fmt::Formatter<'_>,
    shape:   &[usize],
    idx:     &mut Vec<usize>,
    dim:     usize,
    _indent: usize,
) -> std::fmt::Result {
    if shape.is_empty() {
        // Scalar
        let off = buf.layout.byte_offset(idx).map_err(|_| std::fmt::Error)?;
        let v = read_as_f64(unsafe { buf.raw.as_ptr().add(off) }, buf.dtype, buf.dtype.itemsize());
        return write!(f, "{v}");
    }
    if dim == shape.len() {
        let off = buf.layout.byte_offset(idx).map_err(|_| std::fmt::Error)?;
        let v = read_as_f64(unsafe { buf.raw.as_ptr().add(off) }, buf.dtype, buf.dtype.itemsize());
        return write!(f, "{v}");
    }

    let n = shape[dim];
    // Truncate long axes
    const MAX_DISPLAY: usize = 6;
    write!(f, "[")?;
    let show = n.min(MAX_DISPLAY);
    for i in 0..show {
        if i > 0 { write!(f, ", ")?; }
        idx[dim] = i;
        fmt_buffer_data(buf, f, shape, idx, dim + 1, _indent + 1)?;
    }
    if n > MAX_DISPLAY {
        write!(f, ", …({} more)", n - MAX_DISPLAY)?;
    }
    write!(f, "]")
}

// ─── PartialEq ───────────────────────────────────────────────────────────────

/// Byte-level equality.  Two buffers are equal if they have the same dtype,
/// shape, and identical element bytes (in C order).
///
/// For float types, NaN == NaN (bit-pattern equality).
/// Use [`Buffer::allclose`] for IEEE-754-aware approximate comparison.
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        if self.dtype() != other.dtype() || self.shape() != other.shape() {
            return false;
        }
        if Arc::ptr_eq(&self.raw, &other.raw)
            && self.layout.offset() == other.layout.offset()
            && self.strides() == other.strides()
        {
            return true; // literally the same view
        }
        let a = match self.to_contiguous()  { Ok(b) => b, Err(_) => return false };
        let b = match other.to_contiguous() { Ok(b) => b, Err(_) => return false };
        let ab = unsafe { std::slice::from_raw_parts(a.as_ptr(), a.nbytes()) };
        let bb = unsafe { std::slice::from_raw_parts(b.as_ptr(), b.nbytes()) };
        ab == bb
    }
}

// ─── From impls ───────────────────────────────────────────────────────────────

impl From<Vec<f64>> for Buffer {
    fn from(v: Vec<f64>) -> Self { Buffer::from_vec(v).expect("from Vec<f64>") }
}
impl From<Vec<f32>> for Buffer {
    fn from(v: Vec<f32>) -> Self { Buffer::from_vec(v).expect("from Vec<f32>") }
}
impl From<Vec<i32>> for Buffer {
    fn from(v: Vec<i32>) -> Self { Buffer::from_vec(v).expect("from Vec<i32>") }
}
impl From<Vec<i64>> for Buffer {
    fn from(v: Vec<i64>) -> Self { Buffer::from_vec(v).expect("from Vec<i64>") }
}
impl From<Vec<u8>> for Buffer {
    fn from(v: Vec<u8>) -> Self { Buffer::from_vec(v).expect("from Vec<u8>") }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Returns the byte representation of the value "one" for a dtype.
fn dtype_one_bytes(dtype: DType) -> Vec<u8> {
    macro_rules! one_bytes {
        ($T:ty) => {{
            let v = <$T as mohu_dtype::scalar::Scalar>::ONE;
            let size = std::mem::size_of::<$T>();
            let mut bytes = vec![0u8; size];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &v as *const $T as *const u8,
                    bytes.as_mut_ptr(),
                    size,
                );
            }
            bytes
        }};
    }
    use mohu_dtype::dispatch_dtype;
    dispatch_dtype!(dtype, one_bytes)
}

/// Reads a single element at `ptr` (of the given dtype) and casts to f64.
fn read_as_f64(ptr: *const u8, dtype: DType, _itemsize: usize) -> f64 {
    use mohu_dtype::DType::*;
    match dtype {
        Bool => unsafe { *(ptr as *const u8) as f64 },
        I8   => unsafe { *(ptr as *const i8)  as f64 },
        U8   => unsafe { *(ptr as *const u8)  as f64 },
        I16  => unsafe { *(ptr as *const i16) as f64 },
        U16  => unsafe { *(ptr as *const u16) as f64 },
        I32  => unsafe { *(ptr as *const i32) as f64 },
        U32  => unsafe { *(ptr as *const u32) as f64 },
        I64  => unsafe { *(ptr as *const i64) as f64 },
        U64  => unsafe { *(ptr as *const u64) as f64 },
        F32  => unsafe { *(ptr as *const f32)  as f64 },
        F64  => unsafe { *(ptr as *const f64) },
        F16  => {
            let bits = unsafe { (ptr as *const u16).read_unaligned() };
            half::f16::from_bits(bits).to_f64()
        }
        BF16 => {
            let bits = unsafe { (ptr as *const u16).read_unaligned() };
            half::bf16::from_bits(bits).to_f64()
        }
        C64  => unsafe { *(ptr as *const f32) as f64 }, // real part
        C128 => unsafe { *(ptr as *const f64) },         // real part
    }
}

/// Converts a flat C-order index to a multi-dimensional index for `shape`.
fn flat_to_indices(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let nd = shape.len();
    let mut idx = vec![0usize; nd];
    for i in (0..nd).rev() {
        idx[i] = flat % shape[i];
        flat /= shape[i];
    }
    idx
}

use num_traits;
#[cfg(unix)]
use libc;
