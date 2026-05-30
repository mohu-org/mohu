/// Advanced indexing for mohu arrays.
///
/// This crate implements all indexing modes beyond basic scalar / slice
/// indexing, which is handled by `mohu-buffer`'s `Layout::slice_axis`.
///
/// # Indexing modes
///
/// | Module       | Mode                          | Example                    |
/// |--------------|-------------------------------|----------------------------|
/// | [`fancy`]    | Integer array indexing        | `a[[0, 2, 4]]`             |
/// | [`boolean`]  | Boolean mask indexing         | `a[a > 0]`                 |
/// | [`take`]     | `take` / `put` operations     | `take(a, indices, axis=0)` |
/// | [`where_op`] | `where` / `nonzero`           | `np.where(cond, x, y)`     |
/// | [`mod@slice`]| Composite slice objects       | `s_[1:5:2, None, ...]`     |
/// | [`gather`]   | Scatter / gather              | used internally by ufuncs  |
///
/// # Notes on fancy indexing
///
/// Fancy (integer array) indexing always returns a **copy** — this matches
/// NumPy semantics and avoids the aliasing problems that would arise from
/// a view with non-monotonic strides.
///
/// Boolean indexing also returns a copy because the output length is not
/// known until the mask is scanned.
pub mod boolean;
pub mod fancy;
pub mod gather;
pub mod slice;
pub mod take;
pub mod where_op;

pub use mohu_error::{MohuError, MohuResult};
