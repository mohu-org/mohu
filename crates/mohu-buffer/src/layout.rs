/// Array layout descriptor: shape, byte strides, offset, and element size.
///
/// A `Layout` does **not** own any bytes — it merely describes how to index
/// into a backing buffer.  Multiple `Layout`s can describe overlapping views
/// into the same `Buffer` (e.g. a `transpose` creates a new `Layout` with
/// permuted strides, pointing into the same `Arc<RawBuffer>`).
///
/// # Coordinate system
///
/// - `shape[i]` — number of elements along axis `i`.
/// - `strides[i]` — byte distance between consecutive elements along axis `i`.
///   A stride of 0 means this is a broadcast (virtual replication) axis.
/// - `offset` — byte offset from the start of the backing buffer to the
///   element at index `[0, 0, …, 0]`.
/// - `itemsize` — size in bytes of a single scalar element.
use mohu_error::{MohuError, MohuResult};

use crate::strides::{
    self, ShapeVec, StrideVec, broadcast_strides, c_strides, f_strides, validate_strides,
};

// ─── Order ────────────────────────────────────────────────────────────────────

/// Memory order preference for new contiguous allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// Row-major (C) order — the last axis varies fastest in memory.
    C,
    /// Column-major (Fortran) order — the first axis varies fastest.
    F,
}

// ─── SliceArg ────────────────────────────────────────────────────────────────

/// Arguments for slicing a single axis: `start:stop:step`.
///
/// - `None` values default to 0 (start), `dim` (stop), 1 (step).
/// - Negative values wrap around: −1 means the last element.
/// - `step` may be negative for reversed iteration but must not be zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceArg {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

impl SliceArg {
    /// A slice that selects the full axis (`..`).
    pub const FULL: Self = Self {
        start: None,
        stop: None,
        step: None,
    };

    /// Resolves this `SliceArg` against an axis of length `dim`, returning
    /// `(start_index, element_count, step)` in element units.
    ///
    /// Returns `Err(ZeroSliceStep)` if `step == 0`.
    pub fn resolve(self, dim: usize) -> MohuResult<(usize, usize, isize)> {
        let step = self.step.unwrap_or(1);
        if step == 0 {
            return Err(MohuError::ZeroSliceStep);
        }
        let idim = dim as i64;

        // Clamp helper: wraps negatives and clamps to [0, dim].
        let clamp = |v: i64, _default: i64| -> usize {
            let v = if v < 0 { (v + idim).max(0) } else { v };
            v.min(idim) as usize
        };

        let (start, stop) = if step > 0 {
            let s = self.start.map(|v| clamp(v, 0)).unwrap_or(0);
            let e = self.stop.map(|v| clamp(v, idim)).unwrap_or(dim);
            (s, e)
        } else {
            // Reversed: default start = last element, default stop = before start.
            let s = self
                .start
                .map(|v| clamp(v, idim - 1))
                .unwrap_or(dim.saturating_sub(1));
            let e = self
                .stop
                .map(|v| {
                    if v < 0 {
                        ((v + idim).max(-1)) as usize
                    } else {
                        (v as usize).min(dim)
                    }
                })
                .unwrap_or(usize::MAX); // sentinel for "before index 0"
            (s, e)
        };

        let count = if step > 0 {
            if stop <= start {
                0
            } else {
                (stop - start).div_ceil(step as usize)
            }
        } else {
            let abs_step = (-step) as usize;
            if e_reversed_empty(start, stop) {
                0
            } else {
                start.saturating_sub(stop).div_ceil(abs_step)
            }
        };

        Ok((start, count, step as isize))
    }
}

fn e_reversed_empty(start: usize, stop: usize) -> bool {
    // For reversed slices, the stop sentinel is usize::MAX when no stop given.
    stop != usize::MAX && stop >= start
}

// ─── Layout ───────────────────────────────────────────────────────────────────

/// Describes the shape, stride, offset, and element size of an N-dimensional
/// array view.
///
/// Does not own memory — it is always paired with a backing `Buffer`.
#[derive(Clone, PartialEq, Eq)]
pub struct Layout {
    shape: ShapeVec,
    strides: StrideVec,
    /// Byte offset from buffer start to element `[0, 0, …, 0]`.
    offset: usize,
    /// Size in bytes of one scalar element.
    itemsize: usize,
}

impl Layout {
    // ─── Constructors ─────────────────────────────────────────────────────────

    /// Creates a C-contiguous (row-major) layout for `shape`.
    pub fn new_c(shape: &[usize], itemsize: usize) -> MohuResult<Self> {
        let strides = c_strides(shape, itemsize);
        Self::new_custom(shape, &strides, 0, itemsize)
    }

    /// Creates a Fortran-contiguous (column-major) layout for `shape`.
    pub fn new_f(shape: &[usize], itemsize: usize) -> MohuResult<Self> {
        let strides = f_strides(shape, itemsize);
        Self::new_custom(shape, &strides, 0, itemsize)
    }

    /// Creates a layout from explicit shape, strides, offset, and itemsize.
    ///
    /// Validates that:
    /// - `shape.len() == strides.len()`
    /// - All non-broadcast strides are multiples of `itemsize`
    pub fn new_custom(
        shape: &[usize],
        strides: &[isize],
        offset: usize,
        itemsize: usize,
    ) -> MohuResult<Self> {
        if shape.len() != strides.len() {
            return Err(MohuError::bug(format!(
                "Layout::new_custom: shape.len()={} != strides.len()={}",
                shape.len(),
                strides.len()
            )));
        }
        if itemsize == 0 {
            return Err(MohuError::bug("Layout::new_custom: itemsize == 0"));
        }
        validate_strides(shape, strides, itemsize, false)?;
        Ok(Self {
            shape: ShapeVec::from_slice(shape),
            strides: StrideVec::from_slice(strides),
            offset,
            itemsize,
        })
    }

    /// Creates a 0-dimensional (scalar) layout holding exactly one element.
    pub fn scalar(itemsize: usize) -> Self {
        Self {
            shape: ShapeVec::new(),
            strides: StrideVec::new(),
            offset: 0,
            itemsize,
        }
    }

    // ─── Properties ───────────────────────────────────────────────────────────

    /// Number of dimensions (axes).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements (product of shape).
    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// The shape of the array, as a slice of dimension sizes.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The byte strides of the array.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Byte offset of element `[0, 0, …, 0]` from the buffer start.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Size in bytes of a single element.
    #[inline]
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// Total bytes in a contiguous copy of this array (`size * itemsize`).
    ///
    /// Note: for non-contiguous views this may be smaller than the actual
    /// span of bytes touched in the backing buffer.
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize
    }

    /// The actual byte span from the first to the last element, inclusive of
    /// that element's bytes.  For contiguous arrays this equals `nbytes()`.
    /// For strided/non-contiguous views this is larger.
    pub fn span(&self) -> usize {
        if self.size() == 0 {
            return 0;
        }
        let (lo, hi) = self.byte_range();
        hi - lo + self.itemsize
    }

    /// Returns `true` if this is a 0-dimensional (scalar) array.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    /// Returns `true` if any dimension is 0 (zero-element array).
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    // ─── Contiguity checks ────────────────────────────────────────────────────

    /// Returns `true` if the array is C-contiguous (row-major).
    ///
    /// An array is C-contiguous if its strides are `itemsize * product(shape[i+1..])`.
    pub fn is_c_contiguous(&self) -> bool {
        if self.is_empty() || self.ndim() == 0 {
            return true;
        }
        let expected = c_strides(&self.shape, self.itemsize);
        self.strides.as_slice() == expected.as_slice()
    }

    /// Returns `true` if the array is Fortran-contiguous (column-major).
    pub fn is_f_contiguous(&self) -> bool {
        if self.is_empty() || self.ndim() == 0 {
            return true;
        }
        let expected = f_strides(&self.shape, self.itemsize);
        self.strides.as_slice() == expected.as_slice()
    }

    /// Returns `true` if the array is contiguous in either C or F order.
    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous() || self.is_f_contiguous()
    }

    // ─── Transformations ──────────────────────────────────────────────────────

    /// Returns a new `Layout` with axes permuted according to `axes`.
    ///
    /// `axes` must be a permutation of `0..ndim`.
    pub fn permute(&self, axes: &[usize]) -> MohuResult<Self> {
        let ndim = self.ndim();
        if axes.len() != ndim {
            return Err(MohuError::DimensionMismatch {
                expected: ndim,
                got: axes.len(),
            });
        }
        // Validate that `axes` is a proper permutation.
        let mut seen = vec![false; ndim];
        for &ax in axes {
            if ax >= ndim {
                return Err(MohuError::AxisOutOfRange {
                    axis: ax as i64,
                    ndim,
                    valid: format!("0..{ndim}"),
                });
            }
            if seen[ax] {
                return Err(MohuError::bug(format!(
                    "Layout::permute: axis {ax} appears more than once"
                )));
            }
            seen[ax] = true;
        }
        let new_shape: ShapeVec = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: StrideVec = axes.iter().map(|&a| self.strides[a]).collect();
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        })
    }

    /// Returns the transpose — equivalent to NumPy's `.T`.
    ///
    /// Reverses the axis order.  For 2D arrays this transposes the matrix.
    /// For scalars and 1D arrays, returns the layout unchanged.
    pub fn transpose(&self) -> Self {
        if self.ndim() <= 1 {
            return self.clone();
        }
        let axes: Vec<usize> = (0..self.ndim()).rev().collect();
        // Guaranteed to succeed since axes is a valid permutation.
        self.permute(&axes).expect("transpose permute")
    }

    /// Inserts a new axis of size 1 at position `axis`.
    ///
    /// Equivalent to `np.expand_dims(a, axis)`.
    pub fn expand_dims(&self, axis: usize) -> MohuResult<Self> {
        let ndim = self.ndim();
        if axis > ndim {
            return Err(MohuError::AxisOutOfRange {
                axis: axis as i64,
                ndim,
                valid: format!("0..={ndim}"),
            });
        }
        let mut new_shape = ShapeVec::with_capacity(ndim + 1);
        let mut new_strides = StrideVec::with_capacity(ndim + 1);
        for i in 0..=ndim {
            if i == axis {
                new_shape.push(1);
                new_strides.push(self.itemsize as isize); // stride for a size-1 axis
            }
            if i < ndim {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        })
    }

    /// Removes all axes of size 1.
    ///
    /// Equivalent to `np.squeeze(a)`.
    pub fn squeeze(&self) -> Self {
        let new_shape: ShapeVec = self.shape.iter().copied().filter(|&d| d != 1).collect();
        let new_strides: StrideVec = self
            .shape
            .iter()
            .zip(self.strides.iter())
            .filter_map(|(&d, &s)| if d != 1 { Some(s) } else { None })
            .collect();
        Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        }
    }

    /// Removes the size-1 axis at `axis`.
    ///
    /// Returns an error if `axis` is out of range or not a size-1 axis.
    pub fn squeeze_axis(&self, axis: usize) -> MohuResult<Self> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(MohuError::AxisOutOfRange {
                axis: axis as i64,
                ndim,
                valid: format!("0..{ndim}"),
            });
        }
        if self.shape[axis] != 1 {
            return Err(MohuError::bug(format!(
                "squeeze_axis({axis}): axis has size {}, not 1",
                self.shape[axis]
            )));
        }
        let new_shape: ShapeVec = self
            .shape
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if i != axis { Some(d) } else { None })
            .collect();
        let new_strides: StrideVec = self
            .strides
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i != axis { Some(s) } else { None })
            .collect();
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        })
    }

    /// Returns a layout describing a contiguous reshape to `new_shape`.
    ///
    /// Returns `Err(NonContiguous)` if this layout is not C-contiguous,
    /// or `Err(ReshapeIncompatible)` if the element counts differ.
    pub fn reshape(&self, new_shape: &[usize]) -> MohuResult<Self> {
        if !self.is_c_contiguous() {
            return Err(MohuError::NonContiguous);
        }
        let src_len = self.size();
        let dst_len: usize = new_shape.iter().product();
        if src_len != dst_len {
            return Err(MohuError::ReshapeIncompatible {
                src_len,
                dst_shape: new_shape.to_vec(),
                dst_len,
            });
        }
        let new_strides = c_strides(new_shape, self.itemsize);
        Ok(Self {
            shape: ShapeVec::from_slice(new_shape),
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        })
    }

    /// Slices axis `axis` with the given `SliceArg`, returning a new layout that
    /// describes the sub-view without copying data.
    ///
    /// The backing buffer is shared — only the offset, stride, and shape change.
    pub fn slice_axis(&self, axis: usize, arg: SliceArg) -> MohuResult<Self> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(MohuError::AxisOutOfRange {
                axis: axis as i64,
                ndim,
                valid: format!("0..{ndim}"),
            });
        }
        let dim = self.shape[axis];
        let (start, count, step) = arg.resolve(dim)?;

        // Advance the base offset by `start` steps along this axis.
        let new_offset = (self.offset as isize + start as isize * self.strides[axis]) as usize;

        // Multiply the stride by the step size.
        let new_stride = self.strides[axis] * step;

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape[axis] = count;
        new_strides[axis] = new_stride;

        Ok(Self {
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            itemsize: self.itemsize,
        })
    }

    /// Returns a layout representing a broadcast of this array to `new_shape`.
    ///
    /// Broadcast axes get stride 0 — the same element is reused virtually
    /// for every index, without copying.
    pub fn broadcast_to(&self, new_shape: &[usize]) -> MohuResult<Self> {
        let new_strides = broadcast_strides(&self.shape, &self.strides, new_shape)?;
        Ok(Self {
            shape: ShapeVec::from_slice(new_shape),
            strides: new_strides,
            offset: self.offset,
            itemsize: self.itemsize,
        })
    }

    // ─── Indexing ─────────────────────────────────────────────────────────────

    /// Computes the byte offset of element `indices` from the buffer start.
    ///
    /// Returns an error if any index is out of bounds.
    pub fn byte_offset(&self, indices: &[usize]) -> MohuResult<usize> {
        if indices.len() != self.ndim() {
            return Err(MohuError::TooManyIndices {
                given: indices.len(),
                ndim: self.ndim(),
            });
        }
        for (axis, (&idx, &dim)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim {
                return Err(MohuError::IndexOutOfBounds {
                    index: idx as i64,
                    axis,
                    size: dim,
                });
            }
        }
        Ok(self.byte_offset_unchecked(indices))
    }

    /// Computes the byte offset of element `indices` without bounds checking.
    ///
    /// # Safety
    ///
    /// Caller must guarantee `indices[i] < shape[i]` for all `i`.
    #[inline]
    pub fn byte_offset_unchecked(&self, indices: &[usize]) -> usize {
        let off = strides::byte_offset(indices, &self.strides, self.offset);
        off as usize
    }

    // ─── Byte range & overlap detection ──────────────────────────────────────

    /// Returns `(min_byte_offset, max_byte_offset)` — the range of bytes
    /// touched by any element of this layout.
    ///
    /// For empty arrays, returns `(0, 0)`.
    pub fn byte_range(&self) -> (usize, usize) {
        if self.size() == 0 {
            return (self.offset, self.offset);
        }
        let mut lo = self.offset as isize;
        let mut hi = self.offset as isize;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if stride == 0 || dim == 0 {
                continue;
            }
            let span = stride * (dim as isize - 1);
            if span > 0 {
                hi += span;
            } else {
                lo += span;
            }
        }
        (lo as usize, hi as usize)
    }

    /// Returns `true` if this layout's byte range overlaps with `other`'s,
    /// assuming both reference the same backing buffer.
    ///
    /// This is a conservative check — it may return `true` for non-overlapping
    /// stride patterns if their bounding boxes overlap.  Used to detect
    /// aliased mutable access before in-place operations.
    pub fn overlaps_with(&self, other: &Layout) -> bool {
        if self.size() == 0 || other.size() == 0 {
            return false;
        }
        let (lo1, hi1) = self.byte_range();
        let (lo2, hi2) = other.byte_range();
        let hi1 = hi1 + self.itemsize;
        let hi2 = hi2 + other.itemsize;
        lo1 < hi2 && lo2 < hi1
    }

    // ─── Normalisation ────────────────────────────────────────────────────────

    /// Returns a C-contiguous layout with the same shape as `self`,
    /// suitable for a freshly allocated contiguous copy.
    pub fn to_c_contiguous(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: c_strides(&self.shape, self.itemsize),
            offset: 0,
            itemsize: self.itemsize,
        }
    }

    /// Normalises the axis index, supporting negative wrap-around.
    ///
    /// Returns `Err(AxisOutOfRange)` if the index is invalid.
    pub fn normalise_axis(&self, axis: i64) -> MohuResult<usize> {
        let ndim = self.ndim() as i64;
        let a = if axis < 0 { axis + ndim } else { axis };
        if a < 0 || a >= ndim {
            return Err(MohuError::AxisOutOfRange {
                axis,
                ndim: self.ndim(),
                valid: format!("-{ndim}..{ndim}"),
            });
        }
        Ok(a as usize)
    }
}

impl std::fmt::Debug for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layout")
            .field("shape", &self.shape.as_slice())
            .field("strides", &self.strides.as_slice())
            .field("offset", &self.offset)
            .field("itemsize", &self.itemsize)
            .finish()
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout(shape={:?}, strides={:?}, itemsize={})",
            self.shape.as_slice(),
            self.strides.as_slice(),
            self.itemsize
        )
    }
}
