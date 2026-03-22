/// Return early from the current function with a `MohuError`.
///
/// # Examples
///
/// ```rust
/// # use mohu_error::{MohuResult, MohuError, bail};
/// fn check_ndim(ndim: usize) -> MohuResult<()> {
///     if ndim > 32 {
///         bail!(MohuError::DimensionMismatch { expected: 32, got: ndim });
///     }
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! bail {
    ($err:expr) => {
        return ::std::result::Result::Err($err)
    };
}

/// Return early with a `MohuError` if a condition is false.
///
/// # Examples
///
/// ```rust
/// # use mohu_error::{MohuResult, MohuError, ensure};
/// fn divide(a: f64, b: f64) -> MohuResult<f64> {
///     ensure!(b != 0.0, MohuError::DivisionByZero);
///     Ok(a / b)
/// }
/// ```
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $err:expr) => {
        if !($cond) {
            return ::std::result::Result::Err($err);
        }
    };
}

/// Assert that two shapes are equal, returning a [`ShapeMismatch`] error if not.
///
/// [`ShapeMismatch`]: crate::MohuError::ShapeMismatch
///
/// # Examples
///
/// ```rust
/// # use mohu_error::{MohuResult, assert_shape_eq};
/// fn add_shapes(a: &[usize], b: &[usize]) -> MohuResult<()> {
///     assert_shape_eq!(a, b);
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! assert_shape_eq {
    ($lhs:expr, $rhs:expr) => {
        if $lhs != $rhs {
            return ::std::result::Result::Err($crate::MohuError::ShapeMismatch {
                expected: $lhs.to_vec(),
                got: $rhs.to_vec(),
            });
        }
    };
}

/// Assert that an axis index is in range for an array with `ndim` dimensions.
///
/// Accepts both positive and negative axis values and normalises them.
/// Returns [`AxisOutOfRange`] if the axis is out of bounds.
///
/// [`AxisOutOfRange`]: crate::MohuError::AxisOutOfRange
///
/// # Examples
///
/// ```rust
/// # use mohu_error::{MohuResult, assert_axis_valid};
/// fn check(axis: i64, ndim: usize) -> MohuResult<usize> {
///     Ok(assert_axis_valid!(axis, ndim))
/// }
/// ```
#[macro_export]
macro_rules! assert_axis_valid {
    ($axis:expr, $ndim:expr) => {{
        let ax: i64 = $axis;
        let nd: usize = $ndim;
        let nd_i = nd as i64;
        if ax < -nd_i || ax >= nd_i {
            return ::std::result::Result::Err($crate::MohuError::AxisOutOfRange {
                axis: ax,
                ndim: nd,
                valid: if nd == 0 {
                    "none (array is 0-dimensional)".to_string()
                } else {
                    format!("{}..{}", -(nd_i), nd_i - 1)
                },
            });
        }
        if ax < 0 { (nd_i + ax) as usize } else { ax as usize }
    }};
}

/// Assert that an index is in bounds for an axis of given size.
///
/// Normalises negative indices. Returns [`IndexOutOfBounds`] on failure.
///
/// [`IndexOutOfBounds`]: crate::MohuError::IndexOutOfBounds
#[macro_export]
macro_rules! assert_in_bounds {
    ($index:expr, $axis:expr, $size:expr) => {{
        let idx: i64 = $index;
        let sz: usize = $size;
        let sz_i = sz as i64;
        if idx < -sz_i || idx >= sz_i {
            return ::std::result::Result::Err($crate::MohuError::IndexOutOfBounds {
                index: idx,
                axis: $axis,
                size: sz,
            });
        }
        if idx < 0 { (sz_i + idx) as usize } else { idx as usize }
    }};
}
