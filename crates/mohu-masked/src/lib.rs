/// Masked arrays for mohu — null / invalid value propagation.
///
/// A `MaskedArray` is a pair of `(data: Buffer, mask: Buffer<bool>)` where
/// `mask[i] == true` means element `i` is **invalid** (masked out).
///
/// This matches NumPy's `ma` module semantics.
///
/// # Design
///
/// - The mask is stored as a separate `Bool` buffer with the same shape.
/// - Operations propagate masks automatically: `masked op masked → masked`.
/// - Reduction functions skip masked elements by default.
/// - `fill_value` is the value written when converting back to a plain array.
///
/// # Modules
///
/// | Module       | Responsibility                                         |
/// |--------------|--------------------------------------------------------|
/// | [`array`]    | `MaskedArray` type, construction, fill_value           |
/// | [`arith`]    | arithmetic with mask propagation                       |
/// | [`reduce`]   | sum/mean/min/max/std skipping masked elements          |
/// | [`compress`] | `compress`, `compressed` — extract non-masked elements |
/// | [`fill`]     | `filled` — replace masked with fill_value              |
/// | [`mask_ops`] | `masked_where`, `masked_equal`, `getmask`, `getdata`   |
/// | [`io`]       | serialise/deserialise masked arrays (NPY extension)    |
pub mod arith;
pub mod array;
pub mod compress;
pub mod fill;
pub mod io;
pub mod mask_ops;
pub mod reduce;

pub use mohu_error::{MohuError, MohuResult};
