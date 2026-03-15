/// Compile-time and runtime dispatch macros for scalar types.
///
/// These macros are the foundation of how mohu achieves zero-overhead
/// generic dispatch over runtime `DType` values.  Every hot-path kernel
/// in `mohu-buffer`, `mohu-array`, and `mohu-ops` uses `dispatch_dtype!`
/// to monomorphise a single generic function over the correct scalar type
/// without a vtable or allocation.
///
/// # Macro reference
///
/// | Macro | Purpose |
/// |-------|---------|
/// | [`dtype_of!`] | `DType` constant for a Rust type literal |
/// | [`dispatch_dtype!`] | runtime DType → monomorphised call |
/// | [`dispatch_numeric!`] | same, excluding `Bool` |
/// | [`dispatch_integer!`] | integers only |
/// | [`dispatch_float!`] | floats only (F16/BF16/F32/F64) |
/// | [`dispatch_real!`] | integers + real floats (no complex, no bool) |
/// | [`dispatch_signed!`] | signed integers + floats |
/// | [`for_each_dtype!`] | invoke a macro for every dtype (codegen helper) |
/// | [`assert_dtype!`] | assert a DType at runtime or return an error |

// ─── dtype_of! ───────────────────────────────────────────────────────────────

/// Returns the `DType` constant for a Rust primitive type literal.
///
/// This is a purely compile-time macro — it expands to a `DType` constant.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype_of, dtype::DType};
/// assert_eq!(dtype_of!(f32),  DType::F32);
/// assert_eq!(dtype_of!(i64),  DType::I64);
/// assert_eq!(dtype_of!(bool), DType::Bool);
/// ```
#[macro_export]
macro_rules! dtype_of {
    (bool)                        => { $crate::dtype::DType::Bool };
    (i8)                          => { $crate::dtype::DType::I8   };
    (i16)                         => { $crate::dtype::DType::I16  };
    (i32)                         => { $crate::dtype::DType::I32  };
    (i64)                         => { $crate::dtype::DType::I64  };
    (u8)                          => { $crate::dtype::DType::U8   };
    (u16)                         => { $crate::dtype::DType::U16  };
    (u32)                         => { $crate::dtype::DType::U32  };
    (u64)                         => { $crate::dtype::DType::U64  };
    (f16)                         => { $crate::dtype::DType::F16  };
    (::half::f16)                 => { $crate::dtype::DType::F16  };
    (bf16)                        => { $crate::dtype::DType::BF16 };
    (::half::bf16)                => { $crate::dtype::DType::BF16 };
    (f32)                         => { $crate::dtype::DType::F32  };
    (f64)                         => { $crate::dtype::DType::F64  };
    (Complex<f32>)                        => { $crate::dtype::DType::C64  };
    (::num_complex::Complex<f32>)         => { $crate::dtype::DType::C64  };
    (Complex<f64>)                        => { $crate::dtype::DType::C128 };
    (::num_complex::Complex<f64>)         => { $crate::dtype::DType::C128 };
}

// ─── dispatch_dtype! ─────────────────────────────────────────────────────────

/// Dispatches a macro call over all 15 scalar types based on a runtime `DType`.
///
/// Given a `DType` value and the name of a macro, invokes that macro with the
/// corresponding Rust scalar type as a `$ty:ty` argument.  This achieves
/// static monomorphisation at the call site — the compiler generates one
/// specialised version per type and the dispatch is just an integer match.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, dispatch_dtype, scalar::Scalar};
/// fn print_zero(dtype: DType) {
///     macro_rules! print_zero_for {
///         ($T:ty) => { println!("{}", <$T as Scalar>::ZERO) }
///     }
///     dispatch_dtype!(dtype, print_zero_for);
/// }
/// ```
///
/// # Returning values
///
/// The inner macro can expand to an expression, so `dispatch_dtype!` can
/// appear in a `let` binding:
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, dispatch_dtype, scalar::Scalar};
/// fn itemsize(dtype: DType) -> usize {
///     macro_rules! get_size { ($T:ty) => { <$T>::ITEMSIZE } }
///     dispatch_dtype!(dtype, get_size)
/// }
/// ```
#[macro_export]
macro_rules! dispatch_dtype {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::Bool => $macro!(bool),
            $crate::dtype::DType::I8   => $macro!(i8),
            $crate::dtype::DType::I16  => $macro!(i16),
            $crate::dtype::DType::I32  => $macro!(i32),
            $crate::dtype::DType::I64  => $macro!(i64),
            $crate::dtype::DType::U8   => $macro!(u8),
            $crate::dtype::DType::U16  => $macro!(u16),
            $crate::dtype::DType::U32  => $macro!(u32),
            $crate::dtype::DType::U64  => $macro!(u64),
            $crate::dtype::DType::F16  => $macro!(::half::f16),
            $crate::dtype::DType::BF16 => $macro!(::half::bf16),
            $crate::dtype::DType::F32  => $macro!(f32),
            $crate::dtype::DType::F64  => $macro!(f64),
            $crate::dtype::DType::C64  => $macro!(::num_complex::Complex<f32>),
            $crate::dtype::DType::C128 => $macro!(::num_complex::Complex<f64>),
        }
    };
    // Variant that passes extra arguments to the inner macro
    ($dtype:expr, $macro:ident, $($extra:tt)*) => {
        match $dtype {
            $crate::dtype::DType::Bool => $macro!(bool, $($extra)*),
            $crate::dtype::DType::I8   => $macro!(i8,   $($extra)*),
            $crate::dtype::DType::I16  => $macro!(i16,  $($extra)*),
            $crate::dtype::DType::I32  => $macro!(i32,  $($extra)*),
            $crate::dtype::DType::I64  => $macro!(i64,  $($extra)*),
            $crate::dtype::DType::U8   => $macro!(u8,   $($extra)*),
            $crate::dtype::DType::U16  => $macro!(u16,  $($extra)*),
            $crate::dtype::DType::U32  => $macro!(u32,  $($extra)*),
            $crate::dtype::DType::U64  => $macro!(u64,  $($extra)*),
            $crate::dtype::DType::F16  => $macro!(::half::f16,  $($extra)*),
            $crate::dtype::DType::BF16 => $macro!(::half::bf16, $($extra)*),
            $crate::dtype::DType::F32  => $macro!(f32,  $($extra)*),
            $crate::dtype::DType::F64  => $macro!(f64,  $($extra)*),
            $crate::dtype::DType::C64  => $macro!(::num_complex::Complex<f32>, $($extra)*),
            $crate::dtype::DType::C128 => $macro!(::num_complex::Complex<f64>, $($extra)*),
        }
    };
}

// ─── dispatch_numeric! ───────────────────────────────────────────────────────

/// Like `dispatch_dtype!` but excludes `Bool`.
///
/// Returns `Err(UnsupportedDType)` for `DType::Bool`.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, dispatch_numeric};
/// fn sum_zeros(dtype: DType) -> f64 {
///     macro_rules! zero_as_f64 { ($T:ty) => {
///         <$T as mohu_dtype::scalar::Scalar>::ZERO.to_f64_lossy()
///     }}
///     dispatch_numeric!(dtype, zero_as_f64).unwrap()
/// }
/// ```
#[macro_export]
macro_rules! dispatch_numeric {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::Bool => Err($crate::MohuError::UnsupportedDType {
                op: "numeric dispatch",
                dtype: "bool".to_string(),
            }),
            $crate::dtype::DType::I8   => Ok($macro!(i8)),
            $crate::dtype::DType::I16  => Ok($macro!(i16)),
            $crate::dtype::DType::I32  => Ok($macro!(i32)),
            $crate::dtype::DType::I64  => Ok($macro!(i64)),
            $crate::dtype::DType::U8   => Ok($macro!(u8)),
            $crate::dtype::DType::U16  => Ok($macro!(u16)),
            $crate::dtype::DType::U32  => Ok($macro!(u32)),
            $crate::dtype::DType::U64  => Ok($macro!(u64)),
            $crate::dtype::DType::F16  => Ok($macro!(::half::f16)),
            $crate::dtype::DType::BF16 => Ok($macro!(::half::bf16)),
            $crate::dtype::DType::F32  => Ok($macro!(f32)),
            $crate::dtype::DType::F64  => Ok($macro!(f64)),
            $crate::dtype::DType::C64  => Ok($macro!(::num_complex::Complex<f32>)),
            $crate::dtype::DType::C128 => Ok($macro!(::num_complex::Complex<f64>)),
        }
    };
}

// ─── dispatch_integer! ───────────────────────────────────────────────────────

/// Dispatches only over integer types (I8..I64, U8..U64).
///
/// Returns `Err(UnsupportedDType)` for all other dtypes.
#[macro_export]
macro_rules! dispatch_integer {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::I8   => Ok($macro!(i8)),
            $crate::dtype::DType::I16  => Ok($macro!(i16)),
            $crate::dtype::DType::I32  => Ok($macro!(i32)),
            $crate::dtype::DType::I64  => Ok($macro!(i64)),
            $crate::dtype::DType::U8   => Ok($macro!(u8)),
            $crate::dtype::DType::U16  => Ok($macro!(u16)),
            $crate::dtype::DType::U32  => Ok($macro!(u32)),
            $crate::dtype::DType::U64  => Ok($macro!(u64)),
            other => Err($crate::MohuError::UnsupportedDType {
                op: "integer dispatch",
                dtype: other.to_string(),
            }),
        }
    };
}

// ─── dispatch_float! ─────────────────────────────────────────────────────────

/// Dispatches only over real floating-point types (F16, BF16, F32, F64).
///
/// Returns `Err(UnsupportedDType)` for all other dtypes.
#[macro_export]
macro_rules! dispatch_float {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::F16  => Ok($macro!(::half::f16)),
            $crate::dtype::DType::BF16 => Ok($macro!(::half::bf16)),
            $crate::dtype::DType::F32  => Ok($macro!(f32)),
            $crate::dtype::DType::F64  => Ok($macro!(f64)),
            other => Err($crate::MohuError::UnsupportedDType {
                op: "float dispatch",
                dtype: other.to_string(),
            }),
        }
    };
}

// ─── dispatch_real! ──────────────────────────────────────────────────────────

/// Dispatches over real (non-complex, non-bool) types: all integers + F16..F64.
///
/// Returns `Err(UnsupportedDType)` for Bool, C64, and C128.
#[macro_export]
macro_rules! dispatch_real {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::I8   => Ok($macro!(i8)),
            $crate::dtype::DType::I16  => Ok($macro!(i16)),
            $crate::dtype::DType::I32  => Ok($macro!(i32)),
            $crate::dtype::DType::I64  => Ok($macro!(i64)),
            $crate::dtype::DType::U8   => Ok($macro!(u8)),
            $crate::dtype::DType::U16  => Ok($macro!(u16)),
            $crate::dtype::DType::U32  => Ok($macro!(u32)),
            $crate::dtype::DType::U64  => Ok($macro!(u64)),
            $crate::dtype::DType::F16  => Ok($macro!(::half::f16)),
            $crate::dtype::DType::BF16 => Ok($macro!(::half::bf16)),
            $crate::dtype::DType::F32  => Ok($macro!(f32)),
            $crate::dtype::DType::F64  => Ok($macro!(f64)),
            other => Err($crate::MohuError::UnsupportedDType {
                op: "real dispatch",
                dtype: other.to_string(),
            }),
        }
    };
}

// ─── dispatch_signed! ────────────────────────────────────────────────────────

/// Dispatches over signed numeric types: I8..I64 and F16..F64.
///
/// Returns `Err(UnsupportedDType)` for unsigned integers, Bool, and complex.
#[macro_export]
macro_rules! dispatch_signed {
    ($dtype:expr, $macro:ident) => {
        match $dtype {
            $crate::dtype::DType::I8   => Ok($macro!(i8)),
            $crate::dtype::DType::I16  => Ok($macro!(i16)),
            $crate::dtype::DType::I32  => Ok($macro!(i32)),
            $crate::dtype::DType::I64  => Ok($macro!(i64)),
            $crate::dtype::DType::F16  => Ok($macro!(::half::f16)),
            $crate::dtype::DType::BF16 => Ok($macro!(::half::bf16)),
            $crate::dtype::DType::F32  => Ok($macro!(f32)),
            $crate::dtype::DType::F64  => Ok($macro!(f64)),
            other => Err($crate::MohuError::UnsupportedDType {
                op: "signed dispatch",
                dtype: other.to_string(),
            }),
        }
    };
}

// ─── for_each_dtype! ─────────────────────────────────────────────────────────

/// Invokes a macro once for every dtype, in definition order.
///
/// Used as a code-generation helper to expand blanket impls,
/// static tables, or test cases over all types.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::for_each_dtype;
/// macro_rules! print_size {
///     ($T:ty, $D:expr) => {
///         println!("{}: {} bytes", $D, std::mem::size_of::<$T>());
///     }
/// }
/// for_each_dtype!(print_size);
/// ```
#[macro_export]
macro_rules! for_each_dtype {
    ($macro:ident) => {
        $macro!(bool,                           $crate::dtype::DType::Bool);
        $macro!(i8,                             $crate::dtype::DType::I8);
        $macro!(i16,                            $crate::dtype::DType::I16);
        $macro!(i32,                            $crate::dtype::DType::I32);
        $macro!(i64,                            $crate::dtype::DType::I64);
        $macro!(u8,                             $crate::dtype::DType::U8);
        $macro!(u16,                            $crate::dtype::DType::U16);
        $macro!(u32,                            $crate::dtype::DType::U32);
        $macro!(u64,                            $crate::dtype::DType::U64);
        $macro!(::half::f16,                    $crate::dtype::DType::F16);
        $macro!(::half::bf16,                   $crate::dtype::DType::BF16);
        $macro!(f32,                            $crate::dtype::DType::F32);
        $macro!(f64,                            $crate::dtype::DType::F64);
        $macro!(::num_complex::Complex<f32>,    $crate::dtype::DType::C64);
        $macro!(::num_complex::Complex<f64>,    $crate::dtype::DType::C128);
    };
}

// ─── assert_dtype! ───────────────────────────────────────────────────────────

/// Asserts that a runtime `DType` matches the expected value, returning an
/// error with a descriptive message if it does not.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, assert_dtype, MohuResult};
/// fn only_float32(dtype: DType) -> MohuResult<()> {
///     assert_dtype!(dtype, DType::F32, "matrix_mul");
///     Ok(())
/// }
/// ```
#[macro_export]
macro_rules! assert_dtype {
    ($actual:expr, $expected:expr, $op:expr) => {
        if $actual != $expected {
            return Err($crate::MohuError::DTypeMismatch {
                expected: $expected.to_string(),
                got:      $actual.to_string(),
            });
        }
    };
}

// ─── require_float! ──────────────────────────────────────────────────────────

/// Returns an error if the dtype is not a floating-point type.
#[macro_export]
macro_rules! require_float {
    ($dtype:expr, $op:expr) => {
        if !$dtype.is_float() && !$dtype.is_complex() {
            return Err($crate::MohuError::UnsupportedDType {
                op:    $op,
                dtype: $dtype.to_string(),
            });
        }
    };
}

// ─── require_numeric! ────────────────────────────────────────────────────────

/// Returns an error if the dtype is Bool.
#[macro_export]
macro_rules! require_numeric {
    ($dtype:expr, $op:expr) => {
        if $dtype.is_bool() {
            return Err($crate::MohuError::UnsupportedDType {
                op:    $op,
                dtype: "bool".to_string(),
            });
        }
    };
}

// ─── require_real! ───────────────────────────────────────────────────────────

/// Returns an error if the dtype is complex or bool.
#[macro_export]
macro_rules! require_real {
    ($dtype:expr, $op:expr) => {
        if $dtype.is_complex() || $dtype.is_bool() {
            return Err($crate::MohuError::UnsupportedDType {
                op:    $op,
                dtype: $dtype.to_string(),
            });
        }
    };
}
