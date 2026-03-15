/// Scalar-level casting utilities.
///
/// This module implements casts between pairs of concrete scalar types with
/// explicit handling of every edge case that NumPy documents:
///
/// - **Float → Integer**: truncate towards zero, NaN → 0, +Inf → `MAX`,
///   −Inf → `MIN` (NumPy "unsafe" semantics).
/// - **Integer → Float**: may lose precision for wide integers (I64/U64 → F32).
/// - **Float → narrower Float**: round-to-nearest-even via the `half` crate.
/// - **Complex → Real**: takes the real component; imaginary part is discarded.
/// - **Bool → anything**: false=0, true=1.
/// - **anything → Bool**: `x != 0`.
///
/// # Public API
///
/// - [`cast_scalar`] — cast a single scalar value with mode checking.
/// - [`cast_slice`] — cast a contiguous slice of values into a pre-allocated
///   output slice.
use mohu_error::{MohuError, MohuResult};

use crate::{
    promote::CastMode,
    scalar::Scalar,
};

// ─── cast_scalar ─────────────────────────────────────────────────────────────

/// Casts a single scalar value from type `S` to type `D`.
///
/// Checks that the cast is permitted under `mode` before performing it.
///
/// ```rust
/// # use mohu_dtype::{promote::CastMode, cast::cast_scalar};
/// let v: i32 = cast_scalar::<f64, i32>(3.7, CastMode::Unsafe).unwrap();
/// assert_eq!(v, 3);  // truncated towards zero
///
/// let v: f64 = cast_scalar::<i16, f64>(42_i16, CastMode::Safe).unwrap();
/// assert_eq!(v, 42.0);
/// ```
pub fn cast_scalar<S: Scalar, D: Scalar>(value: S, mode: CastMode) -> MohuResult<D> {
    if !crate::promote::can_cast(S::DTYPE, D::DTYPE, mode) {
        return Err(MohuError::InvalidCast {
            from:   S::DTYPE.to_string(),
            to:     D::DTYPE.to_string(),
            reason: format!("{mode:?} cast is not allowed"),
        });
    }
    Ok(cast_scalar_unchecked::<S, D>(value))
}

/// Casts a single scalar without checking cast mode.
///
/// The caller must have already verified that the cast is valid.
/// This is the hot-path function used by kernel loops.
///
/// # Safety (logical)
///
/// This function is safe in the Rust memory safety sense — no unsafe code.
/// "Unchecked" means no mode validation, not no bounds checking.
#[inline(always)]
pub fn cast_scalar_unchecked<S: Scalar, D: Scalar>(value: S) -> D {
    // Optimised fast path: identity cast (S == D at runtime)
    if S::DTYPE == D::DTYPE {
        // SAFETY: If S::DTYPE == D::DTYPE then S and D have the same repr.
        // We use a byte-copy instead of transmute to avoid generic restrictions.
        return byte_copy_cast(value);
    }
    // General path: go through a typed intermediate dispatch.
    cast_via_intermediate(value)
}

// ─── cast_slice ──────────────────────────────────────────────────────────────

/// Casts every element of `src` into `dst`.
///
/// `dst.len()` must equal `src.len()`; returns an error otherwise.
pub fn cast_slice<S: Scalar, D: Scalar>(
    src: &[S],
    dst: &mut [D],
    mode: CastMode,
) -> MohuResult<()> {
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: vec![src.len()],
            got:      vec![dst.len()],
        });
    }
    // Check mode once before the loop.
    if !crate::promote::can_cast(S::DTYPE, D::DTYPE, mode) {
        return Err(MohuError::InvalidCast {
            from:   S::DTYPE.to_string(),
            to:     D::DTYPE.to_string(),
            reason: format!("{mode:?} cast is not allowed"),
        });
    }
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = cast_scalar_unchecked::<S, D>(*s);
    }
    Ok(())
}

// ─── identity byte-copy ───────────────────────────────────────────────────────

#[inline(always)]
fn byte_copy_cast<S: Scalar, D: Scalar>(value: S) -> D {
    debug_assert_eq!(S::ITEMSIZE, D::ITEMSIZE);
    // We can't use `transmute` in generic code directly, so we copy through
    // a stack buffer.  For identity casts this compiles to a no-op.
    let mut buf = [0u8; 16];
    // SAFETY: `value` is a valid `S` on the stack; reading its bytes is sound.
    // `D::ITEMSIZE == S::ITEMSIZE` (asserted in debug builds above), so the
    // ptr::read_unaligned reads exactly the right number of bytes.
    unsafe {
        let bytes = std::slice::from_raw_parts(
            &value as *const S as *const u8,
            S::ITEMSIZE,
        );
        buf[..S::ITEMSIZE].copy_from_slice(bytes);
        std::ptr::read_unaligned(buf.as_ptr() as *const D)
    }
}

// ─── general intermediate dispatch ───────────────────────────────────────────

/// Routes a scalar cast through a runtime-dispatched intermediate representation.
///
/// We use `f64` as the universal lossless-ish intermediate for all non-complex
/// types, and `Complex<f64>` for complex types.
#[inline]
fn cast_via_intermediate<S: Scalar, D: Scalar>(value: S) -> D {
    // Complex → any: take real part, cast that.
    // Any → complex: cast to real, wrap in (real, 0).
    // General: via f64 intermediate.
    D::from_f64_lossy(value.to_f64_lossy())
}

// ─── concrete cast helpers used by dispatch ──────────────────────────────────

/// Cast `f64` to an integer type with NumPy saturation semantics:
/// - NaN     → 0
/// - +Inf    → D::MAX
/// - -Inf    → D::MIN
/// - finite  → round towards zero (truncate), then clamp to [MIN, MAX]
#[inline]
pub fn f64_to_int_saturating<D>(v: f64) -> D
where
    D: Scalar + crate::scalar::RealScalar + crate::scalar::IntScalar,
{
    if v.is_nan() {
        return D::ZERO;
    }
    let min_f = D::min_value().to_f64_lossy();
    let max_f = D::max_value().to_f64_lossy();
    if v <= min_f { return D::min_value(); }
    if v >= max_f { return D::max_value(); }
    D::from_f64_lossy(v)
}

// ─── CastMode std::fmt ───────────────────────────────────────────────────────

impl std::fmt::Display for CastMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Safe     => write!(f, "safe"),
            Self::SameKind => write!(f, "same_kind"),
            Self::Unsafe   => write!(f, "unsafe"),
        }
    }
}
