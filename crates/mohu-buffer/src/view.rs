/// Typed, lifetime-bound views over a `Buffer`.
///
/// `BufferView<'buf, T>` and `BufferViewMut<'buf, T>` are thin wrappers that
/// give safe, typed access to the raw bytes of a `Buffer` without taking
/// ownership.  They are the primary way to read and write individual elements
/// from within hot-path kernel code.
///
/// # Contiguity
///
/// For C-contiguous arrays, views expose `as_slice` / `as_mut_slice` which
/// return a regular Rust slice — zero overhead, compiler can auto-vectorize.
/// For non-contiguous arrays (transpose, slice with step), views fall back to
/// `StridedByteIter`-based element-at-a-time access.
use std::marker::PhantomData;

use mohu_dtype::{dtype::DType, scalar::Scalar};
use mohu_error::{MohuError, MohuResult};

use crate::{
    buffer::Buffer,
    strides::StridedByteIter,
};

// ─── BufferView<'buf, T> ──────────────────────────────────────────────────────

/// An immutable typed view into a `Buffer`.
///
/// # Lifetime
///
/// The view borrows `'buf` from the `Buffer`.  The `Buffer` (and its backing
/// `Arc<RawBuffer>`) must outlive this view.
pub struct BufferView<'buf, T: Scalar> {
    buf:     &'buf Buffer,
    _marker: PhantomData<&'buf T>,
}

impl<'buf, T: Scalar> BufferView<'buf, T> {
    /// Creates a view over `buf`.
    ///
    /// Returns `Err(DTypeMismatch)` if `T::DTYPE != buf.dtype()`.
    pub fn new(buf: &'buf Buffer) -> MohuResult<Self> {
        if T::DTYPE != buf.dtype() {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      buf.dtype().to_string(),
            });
        }
        Ok(Self { buf, _marker: PhantomData })
    }

    // ─── Properties ───────────────────────────────────────────────────────────

    /// Returns the element data type.
    #[inline] pub fn dtype(&self)    -> DType  { self.buf.dtype() }
    /// Returns the number of dimensions.
    #[inline] pub fn ndim(&self)     -> usize  { self.buf.ndim() }
    /// Returns the shape as a slice of dimension sizes.
    #[inline] pub fn shape(&self)    -> &[usize] { self.buf.shape() }
    /// Returns the byte strides.
    #[inline] pub fn strides(&self)  -> &[isize] { self.buf.strides() }
    /// Returns the total number of elements.
    #[inline] pub fn len(&self)      -> usize  { self.buf.len() }
    /// Returns `true` if the buffer has zero elements.
    #[inline] pub fn is_empty(&self) -> bool   { self.buf.is_empty() }
    /// Returns `true` if the buffer is C-contiguous.
    #[inline] pub fn is_c_contiguous(&self) -> bool { self.buf.is_c_contiguous() }

    // ─── Slice access (contiguous only) ───────────────────────────────────────

    /// Returns the data as a flat slice.
    ///
    /// Only valid for C-contiguous buffers.  Returns `Err(NonContiguous)` otherwise.
    pub fn as_slice(&self) -> MohuResult<&'buf [T]> {
        self.buf.as_slice::<T>()
    }

    // ─── Element access ───────────────────────────────────────────────────────

    /// Returns a reference to element `indices`.
    ///
    /// Bounds-checks all indices and walks the stride array.
    /// For hot-loop access in contiguous arrays, use `as_slice` instead.
    pub fn get(&self, indices: &[usize]) -> MohuResult<&T> {
        let off = self.buf.layout().byte_offset(indices)?;
        let ptr = unsafe { self.buf.as_ptr().add(off) as *const T };
        // SAFETY: byte_offset guarantees the offset is within the buffer.
        Ok(unsafe { &*ptr })
    }

    /// Returns a reference to element `indices` without bounds checking.
    ///
    /// # Safety
    ///
    /// `indices[i] < shape[i]` must hold for every `i`.
    #[inline]
    pub unsafe fn get_unchecked(&self, indices: &[usize]) -> &T {
        let off = self.buf.layout().byte_offset_unchecked(indices);
        unsafe { &*(self.buf.as_ptr().add(off) as *const T) }
    }

    // ─── Iterators ────────────────────────────────────────────────────────────

    /// Returns an iterator over references to all elements in C order.
    ///
    /// For contiguous arrays this is a tight slice iterator.
    /// For non-contiguous arrays it uses stride-based byte-offset iteration.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        ViewIter {
            raw_ptr:  self.buf.as_ptr(),
            iter:     StridedByteIter::new(
                self.buf.shape(),
                self.buf.strides(),
                self.buf.offset(),
            ),
            _phantom: PhantomData,
        }
    }
}

// ─── BufferViewMut<'buf, T> ───────────────────────────────────────────────────

/// A mutable typed view into a `Buffer`.
///
/// Acquiring a `BufferViewMut` triggers copy-on-write via
/// [`Buffer::make_unique`] — if the backing bytes are shared, they are
/// copied first.
pub struct BufferViewMut<'buf, T: Scalar> {
    buf:     &'buf mut Buffer,
    _marker: PhantomData<&'buf mut T>,
}

impl<'buf, T: Scalar> BufferViewMut<'buf, T> {
    /// Creates a mutable view over `buf`, triggering CoW if necessary.
    ///
    /// Returns `Err(DTypeMismatch)` if types mismatch, or `Err(ReadOnly)`
    /// if the buffer is marked read-only and cannot be made writeable.
    pub fn new(buf: &'buf mut Buffer) -> MohuResult<Self> {
        if T::DTYPE != buf.dtype() {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      buf.dtype().to_string(),
            });
        }
        if !buf.is_writeable() {
            return Err(MohuError::ReadOnly);
        }
        buf.make_unique()?;
        Ok(Self { buf, _marker: PhantomData })
    }

    // ─── Properties ───────────────────────────────────────────────────────────

    /// Returns the element data type.
    #[inline] pub fn dtype(&self)    -> DType  { self.buf.dtype() }
    /// Returns the number of dimensions.
    #[inline] pub fn ndim(&self)     -> usize  { self.buf.ndim() }
    /// Returns the shape as a slice of dimension sizes.
    #[inline] pub fn shape(&self)    -> &[usize] { self.buf.shape() }
    /// Returns the total number of elements.
    #[inline] pub fn len(&self)      -> usize  { self.buf.len() }
    /// Returns `true` if the buffer has zero elements.
    #[inline] pub fn is_empty(&self) -> bool   { self.buf.is_empty() }

    // ─── Slice access ─────────────────────────────────────────────────────────

    /// Returns the data as a mutable flat slice (contiguous only).
    pub fn as_mut_slice(&mut self) -> MohuResult<&mut [T]> {
        self.buf.as_mut_slice::<T>()
    }

    // ─── Element access ───────────────────────────────────────────────────────

    /// Returns a mutable reference to element `indices`.
    pub fn get_mut(&mut self, indices: &[usize]) -> MohuResult<&mut T> {
        let off = self.buf.layout().byte_offset(indices)?;
        let ptr = unsafe { self.buf.as_mut_ptr().add(off) as *mut T };
        Ok(unsafe { &mut *ptr })
    }

    /// Sets element `indices` to `value`.
    pub fn set(&mut self, indices: &[usize], value: T) -> MohuResult<()> {
        let off = self.buf.layout().byte_offset(indices)?;
        unsafe {
            let ptr = self.buf.as_mut_ptr().add(off) as *mut T;
            ptr.write_unaligned(value);
        }
        Ok(())
    }

    /// Fills every element of the buffer with `value` using Rayon parallelism.
    pub fn fill(&mut self, value: T) -> MohuResult<()>
    where
        T: Copy + Send + Sync,
    {
        crate::ops::fill(self.buf, value)
    }

    // ─── Iterators ────────────────────────────────────────────────────────────

    /// Returns an iterator over mutable references to all elements in C order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        let raw_ptr = unsafe { self.buf.as_mut_ptr() };
        let shape   = self.buf.shape().to_vec();
        let strides = self.buf.strides().to_vec();
        let offset  = self.buf.offset();
        ViewIterMut {
            raw_ptr,
            iter: StridedByteIter::new(&shape, &strides, offset),
            _phantom: PhantomData,
        }
    }
}

// ─── ViewIter ────────────────────────────────────────────────────────────────

struct ViewIter<'a, T> {
    raw_ptr:  *const u8,
    iter:     StridedByteIter,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Scalar> Iterator for ViewIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let off = self.iter.next()?;
        // SAFETY: StridedByteIter offsets are within the buffer's valid range.
        Some(unsafe { &*(self.raw_ptr.add(off) as *const T) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

// ─── ViewIterMut ─────────────────────────────────────────────────────────────

struct ViewIterMut<'a, T> {
    raw_ptr:  *mut u8,
    iter:     StridedByteIter,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Scalar> Iterator for ViewIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        let off = self.iter.next()?;
        // SAFETY: exclusive mutable access is guaranteed by BufferViewMut::new
        // which calls make_unique and checks is_writeable.
        Some(unsafe { &mut *(self.raw_ptr.add(off) as *mut T) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
