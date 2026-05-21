use mohu_core::mohu_buffer::ops::parallel_zip;
use mohu_core::mohu_buffer::Buffer;
use mohu_core::mohu_dtype::{dispatch_dtype, dispatch_real, DType};
use mohu_core::mohu_error::{MohuError, MohuResult};

use crate::broadcast::broadcast_binary_inputs;

#[derive(Clone, Copy)]
enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpKind {
    fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }

    fn is_ordered(self) -> bool {
        matches!(self, Self::Lt | Self::Le | Self::Gt | Self::Ge)
    }
}

fn compare_binary(lhs: &Buffer, rhs: &Buffer, kind: CmpKind) -> MohuResult<Buffer> {
    if lhs.dtype() != rhs.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: lhs.dtype().to_string(),
            got:      rhs.dtype().to_string(),
        });
    }
    if kind.is_ordered() && lhs.dtype().is_complex() {
        return Err(MohuError::domain(
            kind.name(),
            "ordered comparisons are undefined for complex dtypes",
        ));
    }

    let (lhs_buf, rhs_buf, out_shape) = broadcast_binary_inputs(lhs, rhs)?;
    let mut out = Buffer::zeros(DType::Bool, &out_shape)?;

    match kind {
        CmpKind::Eq => {
            macro_rules! do_cmp {
                ($T:ty) => {
                    parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a == b)
                };
            }
            dispatch_dtype!(lhs.dtype(), do_cmp)?;
        }
        CmpKind::Ne => {
            macro_rules! do_cmp {
                ($T:ty) => {
                    parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a != b)
                };
            }
            dispatch_dtype!(lhs.dtype(), do_cmp)?;
        }
        CmpKind::Lt => {
            if lhs.dtype() == DType::Bool {
                parallel_zip::<bool, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a < b)?;
            } else {
                macro_rules! do_cmp {
                    ($T:ty) => {
                        parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a < b)
                    };
                }
                dispatch_real!(lhs.dtype(), do_cmp)??;
            }
        }
        CmpKind::Le => {
            if lhs.dtype() == DType::Bool {
                parallel_zip::<bool, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a <= b)?;
            } else {
                macro_rules! do_cmp {
                    ($T:ty) => {
                        parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a <= b)
                    };
                }
                dispatch_real!(lhs.dtype(), do_cmp)??;
            }
        }
        CmpKind::Gt => {
            if lhs.dtype() == DType::Bool {
                parallel_zip::<bool, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a > b)?;
            } else {
                macro_rules! do_cmp {
                    ($T:ty) => {
                        parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a > b)
                    };
                }
                dispatch_real!(lhs.dtype(), do_cmp)??;
            }
        }
        CmpKind::Ge => {
            if lhs.dtype() == DType::Bool {
                parallel_zip::<bool, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a >= b)?;
            } else {
                macro_rules! do_cmp {
                    ($T:ty) => {
                        parallel_zip::<$T, bool, _>(&lhs_buf, &rhs_buf, &mut out, |a, b| a >= b)
                    };
                }
                dispatch_real!(lhs.dtype(), do_cmp)??;
            }
        }
    }

    Ok(out)
}

/// Element-wise equality comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::eq;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[1_i32, 0, 3])?;
/// let out = eq(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![true, false, true]);
/// # Ok(())
/// # }
/// ```
pub fn eq(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Eq)
}

/// Element-wise inequality comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::ne;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[1_i32, 0, 3])?;
/// let out = ne(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![false, true, false]);
/// # Ok(())
/// # }
/// ```
pub fn ne(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Ne)
}

/// Element-wise less-than comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::lt;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[2_i32, 2, 1])?;
/// let out = lt(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![true, false, false]);
/// # Ok(())
/// # }
/// ```
pub fn lt(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Lt)
}

/// Element-wise less-than-or-equal comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::le;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[1_i32, 1, 4])?;
/// let out = le(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![true, false, true]);
/// # Ok(())
/// # }
/// ```
pub fn le(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Le)
}

/// Element-wise greater-than comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::gt;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[0_i32, 2, 4])?;
/// let out = gt(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![true, false, false]);
/// # Ok(())
/// # }
/// ```
pub fn gt(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Gt)
}

/// Element-wise greater-than-or-equal comparison.
///
/// The output dtype is always `Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_core::mohu_buffer::Buffer;
/// # use mohu_core::mohu_error::MohuResult;
/// # use mohu_ops::cmp::ge;
/// # fn main() -> MohuResult<()> {
/// let a = Buffer::from_slice(&[1_i32, 2, 3])?;
/// let b = Buffer::from_slice(&[1_i32, 3, 3])?;
/// let out = ge(&a, &b)?;
/// assert_eq!(out.to_vec::<bool>()?, vec![true, false, true]);
/// # Ok(())
/// # }
/// ```
pub fn ge(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    compare_binary(lhs, rhs, CmpKind::Ge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mohu_core::mohu_error::MohuError;
    use num_complex::Complex;

    #[test]
    fn eq_broadcasts_and_returns_bool() {
        let lhs = Buffer::from_slice(&[1_i32, 2, 3]).unwrap().reshape(&[1, 3]).unwrap();
        let rhs = Buffer::from_slice(&[1_i32, 0, 3, 1, 2, 4]).unwrap()
            .reshape(&[2, 3]).unwrap();

        let out = eq(&lhs, &rhs).unwrap();
        assert_eq!(out.dtype(), DType::Bool);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(
            out.to_vec::<bool>().unwrap(),
            vec![true, false, true, true, true, false]
        );
    }

    #[test]
    fn nan_semantics_follow_ieee_754() {
        let lhs = Buffer::from_slice(&[f32::NAN, 1.0_f32]).unwrap();
        let rhs = Buffer::from_slice(&[f32::NAN, 2.0_f32]).unwrap();

        let out_eq = eq(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();
        let out_ne = ne(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();
        let out_lt = lt(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();
        let out_le = le(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();
        let out_gt = gt(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();
        let out_ge = ge(&lhs, &rhs).unwrap().to_vec::<bool>().unwrap();

        assert_eq!(out_eq[0], false);
        assert_eq!(out_ne[0], true);
        assert_eq!(out_lt[0], false);
        assert_eq!(out_le[0], false);
        assert_eq!(out_gt[0], false);
        assert_eq!(out_ge[0], false);

        assert_eq!(out_lt[1], true);
        assert_eq!(out_le[1], true);
        assert_eq!(out_gt[1], false);
        assert_eq!(out_ge[1], false);
    }

    #[test]
    fn ordered_comparison_on_complex_is_domain_error() {
        let lhs = Buffer::from_slice(&[Complex::new(1.0_f32, 0.0)]).unwrap();
        let rhs = Buffer::from_slice(&[Complex::new(2.0_f32, 0.0)]).unwrap();

        let err = lt(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, MohuError::DomainError { .. }));
    }

    #[test]
    fn dtype_mismatch_is_error() {
        let lhs = Buffer::from_slice(&[1_i32, 2, 3]).unwrap();
        let rhs = Buffer::from_slice(&[1_f32, 2.0, 3.0]).unwrap();

        let err = eq(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, MohuError::DTypeMismatch { .. }));
    }

    #[test]
    fn broadcast_mismatch_is_shape_error() {
        let lhs = Buffer::from_slice(&[1_i32, 2, 3, 4]).unwrap().reshape(&[2, 2]).unwrap();
        let rhs = Buffer::from_slice(&[1_i32, 2, 3]).unwrap().reshape(&[3]).unwrap();

        let err = eq(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, MohuError::ShapeMismatch { .. }));
    }
}