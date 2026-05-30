/// Stride arithmetic, N-dimensional index iteration, and broadcast utilities.
///
/// Strides are always in **bytes**, not element counts.  For a C-contiguous
/// array with `itemsize = 4` (f32) and `shape = [3, 4]`:
///
/// ```text
/// strides[0] = 4 * 4 = 16   (jump 16 bytes to go to the next row)
/// strides[1] = 4 * 1 = 4    (jump  4 bytes to go to the next column)
/// ```
///
/// A stride of 0 is legal and means that axis is a *broadcast* axis — the
/// same element is reused for every index along that dimension.
use smallvec::SmallVec;

use mohu_error::{MohuError, MohuResult};

// ─── Type aliases ─────────────────────────────────────────────────────────────

/// Stack-allocated shape (avoids heap for ≤ 8 dimensions).
pub type ShapeVec = SmallVec<[usize; 8]>;

/// Stack-allocated stride vector (avoids heap for ≤ 8 dimensions).
pub type StrideVec = SmallVec<[isize; 8]>;

// ─── Stride computation ───────────────────────────────────────────────────────

/// Computes C-contiguous (row-major) byte strides for `shape`.
///
/// The last axis varies fastest.  The stride of axis `i` is:
/// ```text
/// strides[i] = itemsize * product(shape[i+1..])
/// ```
///
/// Returns an empty `StrideVec` for 0-dimensional arrays.
pub fn c_strides(shape: &[usize], itemsize: usize) -> StrideVec {
    let ndim = shape.len();
    let mut strides = StrideVec::with_capacity(ndim);
    let mut acc: isize = itemsize as isize;
    for _ in 0..ndim {
        strides.push(0); // placeholder — filled backwards below
    }
    for i in (0..ndim).rev() {
        strides[i] = acc;
        acc = acc.saturating_mul(shape[i] as isize);
    }
    strides
}

/// Computes Fortran-contiguous (column-major) byte strides for `shape`.
///
/// The first axis varies fastest.  The stride of axis `i` is:
/// ```text
/// strides[i] = itemsize * product(shape[..i])
/// ```
pub fn f_strides(shape: &[usize], itemsize: usize) -> StrideVec {
    let ndim = shape.len();
    let mut strides = StrideVec::with_capacity(ndim);
    let mut acc: isize = itemsize as isize;
    for &s in shape {
        strides.push(acc);
        acc = acc.saturating_mul(s as isize);
    }
    strides
}

/// Computes the total number of elements in `shape`.
///
/// Returns an error on overflow or if the product would exceed `isize::MAX`
/// (which would make byte offsets unrepresentable as `isize`).
pub fn shape_size(shape: &[usize]) -> MohuResult<usize> {
    let mut total: usize = 1;
    for &dim in shape {
        total = total
            .checked_mul(dim)
            .ok_or(MohuError::ShapeOverflow { max: usize::MAX })?;
        if total > isize::MAX as usize {
            return Err(MohuError::ShapeOverflow {
                max: isize::MAX as usize,
            });
        }
    }
    Ok(total)
}

/// Returns `shape_size(shape) * itemsize` bytes required for a contiguous array.
pub fn contiguous_nbytes(shape: &[usize], itemsize: usize) -> MohuResult<usize> {
    if shape.is_empty() {
        // Scalar — one element
        return Ok(itemsize);
    }
    shape_size(shape)?
        .checked_mul(itemsize)
        .ok_or(MohuError::ShapeOverflow { max: usize::MAX })
}

// ─── Broadcast strides ────────────────────────────────────────────────────────

/// Computes broadcast-compatible strides for a source shape/stride pair when
/// broadcasting to a larger `target_shape`.
///
/// Follows NumPy broadcast rules:
/// 1. Leading dimensions are filled with 0-strides (broadcast dimensions).
/// 2. For matching trailing dimensions, the stride is carried over unchanged.
/// 3. For source dimensions of size 1, the stride is set to 0 (broadcast).
/// 4. Dimensions where both source and target are > 1 must be equal.
///
/// Returns `Err(BroadcastError)` if broadcasting is impossible.
pub fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    tgt_shape: &[usize],
) -> MohuResult<StrideVec> {
    let src_ndim = src_shape.len();
    let tgt_ndim = tgt_shape.len();

    if src_ndim > tgt_ndim {
        return Err(MohuError::BroadcastError {
            lhs: src_shape.to_vec(),
            rhs: tgt_shape.to_vec(),
        });
    }

    let mut out = StrideVec::with_capacity(tgt_ndim);

    // Fill leading new dimensions with stride 0.
    let offset = tgt_ndim - src_ndim;
    for _ in 0..offset {
        out.push(0);
    }

    // Align and validate the trailing dimensions.
    for (axis, (&s_dim, &s_stride)) in src_shape.iter().zip(src_strides.iter()).enumerate() {
        let t_dim = tgt_shape[offset + axis];
        if s_dim == t_dim {
            out.push(s_stride);
        } else if s_dim == 1 {
            out.push(0); // broadcast this axis
        } else {
            return Err(MohuError::BroadcastError {
                lhs: src_shape.to_vec(),
                rhs: tgt_shape.to_vec(),
            });
        }
    }

    Ok(out)
}

// ─── Index conversion ─────────────────────────────────────────────────────────

/// Converts a flat linear index into a multi-dimensional index for `shape`.
///
/// Uses C (row-major) order.  Returns an error if `flat >= shape_size(shape)`.
pub fn unravel_index(flat: usize, shape: &[usize]) -> MohuResult<ShapeVec> {
    let size = shape_size(shape)?;
    if flat >= size && size > 0 {
        return Err(MohuError::IndexOutOfBounds {
            index: flat as i64,
            axis: 0,
            size,
        });
    }
    let mut idx = ShapeVec::with_capacity(shape.len());
    let mut rem = flat;
    for i in (0..shape.len()).rev() {
        idx.push(rem % shape[i]);
        rem /= shape[i];
    }
    idx.reverse();
    Ok(idx)
}

/// Converts a multi-dimensional index to a flat C-order linear index.
///
/// Returns an error if any index component is out of bounds.
pub fn ravel_multi_index(indices: &[usize], shape: &[usize]) -> MohuResult<usize> {
    if indices.len() != shape.len() {
        return Err(MohuError::TooManyIndices {
            given: indices.len(),
            ndim: shape.len(),
        });
    }
    let mut flat: usize = 0;
    for (axis, (&idx, &dim)) in indices.iter().zip(shape.iter()).enumerate() {
        if idx >= dim {
            return Err(MohuError::IndexOutOfBounds {
                index: idx as i64,
                axis,
                size: dim,
            });
        }
        flat = flat * dim + idx;
    }
    Ok(flat)
}

/// Computes a byte offset from multi-dimensional indices, strides, and base offset.
///
/// # Safety / correctness
///
/// Caller must guarantee `indices[i] < shape[i]` for all `i`.
#[inline]
pub fn byte_offset(indices: &[usize], strides: &[isize], base_offset: usize) -> isize {
    let mut off: isize = base_offset as isize;
    for (&idx, &stride) in indices.iter().zip(strides.iter()) {
        off += idx as isize * stride;
    }
    off
}

// ─── NdIndexIter ─────────────────────────────────────────────────────────────

/// Iterator over every multi-dimensional index of an N-dimensional array,
/// in C (row-major) order.
///
/// ```rust
/// # use mohu_buffer::strides::NdIndexIter;
/// let iter = NdIndexIter::new(&[2, 3]);
/// let indices: Vec<_> = iter.collect();
/// assert_eq!(indices[0].as_slice(), &[0, 0]);
/// assert_eq!(indices[5].as_slice(), &[1, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct NdIndexIter {
    shape: ShapeVec,
    current: ShapeVec,
    done: bool,
    count: usize,
    total: usize,
}

impl NdIndexIter {
    /// Creates an iterator over all indices of `shape`.
    ///
    /// Returns an empty iterator for zero-sized shapes (any dimension is 0).
    pub fn new(shape: &[usize]) -> Self {
        let total = shape.iter().product::<usize>();
        let current = SmallVec::from_slice(&vec![0usize; shape.len()]);
        let done = total == 0;
        Self {
            shape: ShapeVec::from_slice(shape),
            current,
            done,
            count: 0,
            total,
        }
    }

    /// Returns the total number of indices this iterator will yield.
    pub fn total(&self) -> usize {
        self.total
    }

    fn advance(&mut self) {
        let ndim = self.shape.len();
        if ndim == 0 {
            self.done = true;
            return;
        }
        let mut axis = ndim - 1;
        loop {
            self.current[axis] += 1;
            if self.current[axis] < self.shape[axis] {
                return;
            }
            self.current[axis] = 0;
            if axis == 0 {
                self.done = true;
                return;
            }
            axis -= 1;
        }
    }
}

impl Iterator for NdIndexIter {
    type Item = ShapeVec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let idx = self.current.clone();
        self.count += 1;
        if self.count < self.total {
            self.advance();
        } else {
            self.done = true;
        }
        Some(idx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total.saturating_sub(self.count);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NdIndexIter {}

// ─── StridedByteIter ─────────────────────────────────────────────────────────

/// Iterator that yields the **byte offset** of every element in a strided array,
/// in C order.
///
/// Suitable for implementing element-wise operations on non-contiguous arrays
/// without an allocation.
#[derive(Debug, Clone)]
pub struct StridedByteIter {
    nd_iter: NdIndexIter,
    strides: StrideVec,
    base_offset: usize,
}

impl StridedByteIter {
    /// Creates an iterator that yields byte offsets for a strided array.
    pub fn new(shape: &[usize], strides: &[isize], base_offset: usize) -> Self {
        Self {
            nd_iter: NdIndexIter::new(shape),
            strides: StrideVec::from_slice(strides),
            base_offset,
        }
    }
}

impl Iterator for StridedByteIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.nd_iter.next()?;
        let off = byte_offset(&idx, &self.strides, self.base_offset);
        Some(off as usize)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.nd_iter.size_hint()
    }
}

impl ExactSizeIterator for StridedByteIter {}

// ─── Stride validation ────────────────────────────────────────────────────────

/// Checks that every stride is a multiple of `itemsize`, and that no two
/// distinct elements share the same byte address (overlapping strides on a
/// mutable array would allow unsound aliased mutation).
///
/// Broadcast strides (stride = 0) are excluded from the overlap check because
/// they represent read-only virtual replication of a single element.
pub fn validate_strides(
    shape: &[usize],
    strides: &[isize],
    itemsize: usize,
    mutable: bool,
) -> MohuResult<()> {
    for (axis, (&stride, &dim)) in strides.iter().zip(shape.iter()).enumerate() {
        if dim <= 1 {
            continue; // single-element or broadcast axis — always fine
        }
        // Stride must be a multiple of itemsize (alignment invariant).
        if stride.unsigned_abs() % itemsize != 0 {
            return Err(MohuError::InvalidStride {
                axis,
                stride,
                element_size: itemsize,
            });
        }
    }

    // For mutable arrays, verify no two distinct valid indices map to the same
    // byte offset (which would allow aliased mutable access).
    if mutable {
        check_no_overlap(shape, strides, itemsize)?;
    }

    Ok(())
}

/// Returns `Err(OverlappingStrides)` if the stride+shape combination would
/// cause two distinct elements to share a byte address.
fn check_no_overlap(shape: &[usize], strides: &[isize], itemsize: usize) -> MohuResult<()> {
    // Fast path: only one axis has stride != 0 — can't overlap.
    let non_broadcast: Vec<_> = strides
        .iter()
        .zip(shape.iter())
        .filter(|(s, d)| **s != 0 && **d > 1)
        .collect();
    if non_broadcast.len() <= 1 {
        return Ok(());
    }

    // Slow path: check with a sorted-stride heuristic.
    // Sort axes by |stride| descending; if any axis step is smaller than the
    // entire span of the next axis, we have overlap.
    let mut axes: Vec<(isize, usize)> = strides
        .iter()
        .zip(shape.iter())
        .map(|(&s, &d)| (s, d))
        .filter(|&(s, d)| s != 0 && d > 1)
        .collect();
    axes.sort_by_key(|b| std::cmp::Reverse(b.0.unsigned_abs()));

    for i in 1..axes.len() {
        let (prev_stride, prev_dim) = axes[i - 1];
        let (curr_stride, _) = axes[i];
        // The span of the current axis must not exceed the spacing of the prev.
        if curr_stride.unsigned_abs() < prev_stride.unsigned_abs() * prev_dim {
            return Err(MohuError::OverlappingStrides {
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                element_size: itemsize,
            });
        }
    }

    Ok(())
}
