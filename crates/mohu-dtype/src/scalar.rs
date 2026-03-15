/// The `Scalar` trait hierarchy.
///
/// Every element type that can be stored in a mohu array implements `Scalar`.
/// The trait is **sealed** — only the types defined in this module implement it.
/// This lets mohu make unconditional assumptions about the behaviour of scalar
/// values without worrying about third-party implementations breaking invariants.
///
/// # Hierarchy
///
/// ```text
/// Scalar  (base: Copy, Zero, One, DTYPE, itemsize)
///   ├── RealScalar  (PartialOrd, min/max, abs)
///   │     ├── IntScalar   (bit ops, integer-specific conversions)
///   │     │     ├── SignedScalar   (Neg, overflowing arithmetic)
///   │     │     └── UnsignedScalar
///   │     └── FloatScalar (sqrt, trig, exp/log, NaN/Inf checks)
///   └── ComplexScalar (norm, conj, real/imag split)
/// ```
use num_complex::Complex;

use crate::dtype::DType;

// ─── sealing mechanism ───────────────────────────────────────────────────────

mod private {
    use num_complex::Complex;
    use half::{f16, bf16};

    /// Private marker trait — only types in this module can implement `Scalar`.
    pub trait Sealed {}

    impl Sealed for bool {}
    impl Sealed for i8  {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8  {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f16  {}
    impl Sealed for bf16 {}
    impl Sealed for f32  {}
    impl Sealed for f64  {}
    impl Sealed for Complex<f32> {}
    impl Sealed for Complex<f64> {}
}

// ─── base Scalar trait ───────────────────────────────────────────────────────

/// Base trait for every element type mohu supports.
///
/// Implemented by all 15 scalar types.  Provides only the properties that
/// are universal: DType tag, element size, `Copy`, `Debug`, `Display`.
///
/// `Zero` and `One` are NOT required at the trait level because `bool` does
/// not implement those num-traits. Use the `ZERO` and `ONE` associated
/// constants instead.
pub trait Scalar:
    private::Sealed
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + 'static
    + std::fmt::Debug
    + std::fmt::Display
    + PartialEq
{
    /// The `DType` tag for this scalar type.
    const DTYPE: DType;

    /// Additive identity (same as `Zero::zero()`, provided as a const).
    const ZERO: Self;

    /// Multiplicative identity (same as `One::one()`, provided as a const).
    const ONE: Self;

    /// Size of this type in bytes.
    const ITEMSIZE: usize;

    /// Casts this value to `f64` with best-effort precision.
    /// For complex types, returns the magnitude.
    fn to_f64_lossy(self) -> f64;

    /// Constructs this scalar by casting from `f64`.
    /// For integer types this truncates towards zero.
    /// For half-precision types this rounds to nearest even.
    fn from_f64_lossy(v: f64) -> Self;
}

// ─── RealScalar ─────────────────────────────────────────────────────────────

/// Extension of `Scalar` for types with a natural total order:
/// integers and real floats (not complex, not bool).
pub trait RealScalar: Scalar + PartialOrd {
    /// The minimum finite value of this type.
    fn min_value() -> Self;

    /// The maximum finite value of this type.
    fn max_value() -> Self;

    /// Absolute value.  For unsigned integers this is a no-op.
    fn abs(self) -> Self;

    /// Returns the larger of `self` and `other`.
    fn scalar_max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    /// Returns the smaller of `self` and `other`.
    fn scalar_min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    /// Clamps `self` into `[lo, hi]`.
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.scalar_max(lo).scalar_min(hi)
    }
}

// ─── IntScalar ───────────────────────────────────────────────────────────────

/// Extension for integer scalar types.
pub trait IntScalar:
    RealScalar
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Rem<Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
    + std::ops::Shl<u32, Output = Self>
    + std::ops::Shr<u32, Output = Self>
{
    /// Number of bits in this type.
    const BITS: u32;

    /// Returns `(result, overflowed)` for addition.
    fn overflowing_add(self, rhs: Self) -> (Self, bool);

    /// Returns `(result, overflowed)` for subtraction.
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);

    /// Returns `(result, overflowed)` for multiplication.
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);

    /// Saturating addition — clamps to `[MIN, MAX]` instead of wrapping.
    fn saturating_add(self, rhs: Self) -> Self;

    /// Saturating subtraction.
    fn saturating_sub(self, rhs: Self) -> Self;

    /// Checked addition — returns `None` on overflow.
    fn checked_add(self, rhs: Self) -> Option<Self>;

    /// Checked subtraction — returns `None` on overflow.
    fn checked_sub(self, rhs: Self) -> Option<Self>;

    /// Population count (number of set bits).
    fn count_ones(self) -> u32;

    /// Returns the number of leading zero bits.
    fn leading_zeros(self) -> u32;

    /// Returns the number of trailing zero bits.
    fn trailing_zeros(self) -> u32;

    /// Returns the integer as a `u64` (zero-extended for unsigned,
    /// sign-extended and reinterpreted for signed).
    fn to_u64_bits(self) -> u64;
}

// ─── SignedScalar ─────────────────────────────────────────────────────────────

/// Extension for signed integer types.
pub trait SignedScalar: IntScalar + std::ops::Neg<Output = Self> {
    /// Returns the absolute value, saturating at `Self::max_value()` for
    /// `I8::MIN`, `I16::MIN`, etc.
    fn saturating_abs(self) -> Self;

    /// Returns the sign of `self`: -1, 0, or 1.
    fn signum(self) -> Self;
}

// ─── UnsignedScalar ──────────────────────────────────────────────────────────

/// Marker for unsigned integer types.
pub trait UnsignedScalar: IntScalar {}

// ─── FloatScalar ─────────────────────────────────────────────────────────────

/// Extension for real floating-point types (F16, BF16, F32, F64).
pub trait FloatScalar:
    RealScalar
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// Not-a-number sentinel.
    fn nan() -> Self;

    /// Positive infinity.
    fn infinity() -> Self;

    /// Negative infinity.
    fn neg_infinity() -> Self;

    /// Returns `true` if this value is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if this value is ±infinity.
    fn is_infinite(self) -> bool;

    /// Returns `true` if this value is finite (not NaN, not ±inf).
    fn is_finite(self) -> bool;

    /// Returns `true` if this value is a positive number (not zero, not NaN).
    fn is_sign_positive(self) -> bool;

    /// Returns `true` if this value is a negative number.
    fn is_sign_negative(self) -> bool;

    /// Square root.
    fn sqrt(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Base-2 logarithm.
    fn log2(self) -> Self;

    /// Base-10 logarithm.
    fn log10(self) -> Self;

    /// `e^self`.
    fn exp(self) -> Self;

    /// `2^self`.
    fn exp2(self) -> Self;

    /// Raises self to an integer power.
    fn powi(self, n: i32) -> Self;

    /// Raises self to a floating-point power.
    fn powf(self, n: Self) -> Self;

    /// Floor (round towards -∞).
    fn floor(self) -> Self;

    /// Ceiling (round towards +∞).
    fn ceil(self) -> Self;

    /// Round to nearest integer, ties to even.
    fn round(self) -> Self;

    /// Truncate towards zero.
    fn trunc(self) -> Self;

    /// Fractional part.
    fn fract(self) -> Self;

    /// Fused multiply-add: `(self * a) + b` with a single rounding.
    fn mul_add(self, a: Self, b: Self) -> Self;

    /// Machine epsilon for this type.
    fn epsilon() -> Self;

    /// Smallest positive normalised value.
    fn min_positive() -> Self;

    /// Converts this float to `f32` with best-effort precision.
    fn to_f32(self) -> f32;

    /// Converts this float to `f64`.
    fn to_f64(self) -> f64;
}

// ─── ComplexScalar ────────────────────────────────────────────────────────────

/// Extension for complex scalar types (C64, C128).
pub trait ComplexScalar:
    Scalar
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// The real-part dtype: F32 for C64, F64 for C128.
    type Real: FloatScalar;

    /// Constructs a complex value from real and imaginary parts.
    fn from_re_im(re: Self::Real, im: Self::Real) -> Self;

    /// Returns the real part.
    fn re(self) -> Self::Real;

    /// Returns the imaginary part.
    fn im(self) -> Self::Real;

    /// Complex conjugate.
    fn conj(self) -> Self;

    /// Absolute value (modulus / magnitude).
    fn norm(self) -> Self::Real;

    /// Squared magnitude (faster than `norm()` — no sqrt).
    fn norm_sqr(self) -> Self::Real;

    /// Argument (phase angle) in radians.
    fn arg(self) -> Self::Real;

    /// Returns `true` if either component is NaN.
    fn is_nan(self) -> bool;

    /// Returns `true` if either component is ±infinity.
    fn is_infinite(self) -> bool;

    /// Returns `true` if both components are finite.
    fn is_finite(self) -> bool;
}

// =============================================================================
// Macro-driven implementations
// =============================================================================

// ─── Scalar impls for integers ───────────────────────────────────────────────

macro_rules! impl_scalar_int {
    (
        $ty:ty, $dtype:expr, $zero:expr, $one:expr
    ) => {
        impl Scalar for $ty {
            const DTYPE:    DType  = $dtype;
            const ZERO:     $ty    = $zero;
            const ONE:      $ty    = $one;
            const ITEMSIZE: usize  = std::mem::size_of::<$ty>();

            #[inline] fn to_f64_lossy(self) -> f64    { self as f64 }
            #[inline] fn from_f64_lossy(v: f64) -> $ty { v as $ty }
        }
    };
}

impl_scalar_int!(i8,  DType::I8,  0i8,  1i8);
impl_scalar_int!(i16, DType::I16, 0i16, 1i16);
impl_scalar_int!(i32, DType::I32, 0i32, 1i32);
impl_scalar_int!(i64, DType::I64, 0i64, 1i64);
impl_scalar_int!(u8,  DType::U8,  0u8,  1u8);
impl_scalar_int!(u16, DType::U16, 0u16, 1u16);
impl_scalar_int!(u32, DType::U32, 0u32, 1u32);
impl_scalar_int!(u64, DType::U64, 0u64, 1u64);

// ─── RealScalar for integers ─────────────────────────────────────────────────

impl RealScalar for i8  { fn min_value()->Self{i8::MIN}  fn max_value()->Self{i8::MAX}  fn abs(self)->Self{self.wrapping_abs()} }
impl RealScalar for i16 { fn min_value()->Self{i16::MIN} fn max_value()->Self{i16::MAX} fn abs(self)->Self{self.wrapping_abs()} }
impl RealScalar for i32 { fn min_value()->Self{i32::MIN} fn max_value()->Self{i32::MAX} fn abs(self)->Self{self.wrapping_abs()} }
impl RealScalar for i64 { fn min_value()->Self{i64::MIN} fn max_value()->Self{i64::MAX} fn abs(self)->Self{self.wrapping_abs()} }
impl RealScalar for u8  { fn min_value()->Self{0}        fn max_value()->Self{u8::MAX}  fn abs(self)->Self{self} }
impl RealScalar for u16 { fn min_value()->Self{0}        fn max_value()->Self{u16::MAX} fn abs(self)->Self{self} }
impl RealScalar for u32 { fn min_value()->Self{0}        fn max_value()->Self{u32::MAX} fn abs(self)->Self{self} }
impl RealScalar for u64 { fn min_value()->Self{0}        fn max_value()->Self{u64::MAX} fn abs(self)->Self{self} }

// ─── IntScalar for all integer types ────────────────────────────────────────

macro_rules! impl_int_scalar {
    ($ty:ty) => {
        impl IntScalar for $ty {
            const BITS: u32 = <$ty>::BITS;
            #[inline] fn overflowing_add(self, r: Self) -> (Self, bool) { <$ty>::overflowing_add(self, r) }
            #[inline] fn overflowing_sub(self, r: Self) -> (Self, bool) { <$ty>::overflowing_sub(self, r) }
            #[inline] fn overflowing_mul(self, r: Self) -> (Self, bool) { <$ty>::overflowing_mul(self, r) }
            #[inline] fn saturating_add(self, r: Self) -> Self          { <$ty>::saturating_add(self, r) }
            #[inline] fn saturating_sub(self, r: Self) -> Self          { <$ty>::saturating_sub(self, r) }
            #[inline] fn checked_add(self, r: Self) -> Option<Self>     { <$ty>::checked_add(self, r) }
            #[inline] fn checked_sub(self, r: Self) -> Option<Self>     { <$ty>::checked_sub(self, r) }
            #[inline] fn count_ones(self) -> u32                        { <$ty>::count_ones(self) }
            #[inline] fn leading_zeros(self) -> u32                     { <$ty>::leading_zeros(self) }
            #[inline] fn trailing_zeros(self) -> u32                    { <$ty>::trailing_zeros(self) }
            #[inline] fn to_u64_bits(self) -> u64                       { self as u64 }
        }
    };
}

impl_int_scalar!(i8);
impl_int_scalar!(i16);
impl_int_scalar!(i32);
impl_int_scalar!(i64);
impl_int_scalar!(u8);
impl_int_scalar!(u16);
impl_int_scalar!(u32);
impl_int_scalar!(u64);

// ─── SignedScalar ────────────────────────────────────────────────────────────

macro_rules! impl_signed_scalar {
    ($ty:ty) => {
        impl SignedScalar for $ty {
            #[inline] fn saturating_abs(self) -> Self { <$ty>::saturating_abs(self) }
            #[inline] fn signum(self) -> Self         { <$ty>::signum(self) }
        }
    };
}

impl_signed_scalar!(i8);
impl_signed_scalar!(i16);
impl_signed_scalar!(i32);
impl_signed_scalar!(i64);

// ─── UnsignedScalar ──────────────────────────────────────────────────────────

impl UnsignedScalar for u8  {}
impl UnsignedScalar for u16 {}
impl UnsignedScalar for u32 {}
impl UnsignedScalar for u64 {}

// ─── Bool ────────────────────────────────────────────────────────────────────

impl Scalar for bool {
    const DTYPE:    DType = DType::Bool;
    const ZERO:     bool  = false;
    const ONE:      bool  = true;
    const ITEMSIZE: usize = 1;

    #[inline] fn to_f64_lossy(self) -> f64    { self as u8 as f64 }
    #[inline] fn from_f64_lossy(v: f64) -> bool { v != 0.0 }
}

// ─── Scalar for native floats ─────────────────────────────────────────────────

impl Scalar for f32 {
    const DTYPE:    DType = DType::F32;
    const ZERO:     f32   = 0.0_f32;
    const ONE:      f32   = 1.0_f32;
    const ITEMSIZE: usize = 4;

    #[inline] fn to_f64_lossy(self) -> f64    { self as f64 }
    #[inline] fn from_f64_lossy(v: f64) -> f32 { v as f32 }
}

impl Scalar for f64 {
    const DTYPE:    DType = DType::F64;
    const ZERO:     f64   = 0.0_f64;
    const ONE:      f64   = 1.0_f64;
    const ITEMSIZE: usize = 8;

    #[inline] fn to_f64_lossy(self) -> f64    { self }
    #[inline] fn from_f64_lossy(v: f64) -> f64 { v }
}

// ─── Scalar for half-precision types ─────────────────────────────────────────

impl Scalar for half::f16 {
    const DTYPE:    DType    = DType::F16;
    const ZERO:     half::f16 = half::f16::ZERO;
    const ONE:      half::f16 = half::f16::ONE;
    const ITEMSIZE: usize    = 2;

    #[inline] fn to_f64_lossy(self) -> f64         { self.to_f64() }
    #[inline] fn from_f64_lossy(v: f64) -> half::f16 { half::f16::from_f64(v) }
}

impl Scalar for half::bf16 {
    const DTYPE:    DType     = DType::BF16;
    const ZERO:     half::bf16 = half::bf16::ZERO;
    const ONE:      half::bf16 = half::bf16::ONE;
    const ITEMSIZE: usize     = 2;

    #[inline] fn to_f64_lossy(self) -> f64          { self.to_f64() }
    #[inline] fn from_f64_lossy(v: f64) -> half::bf16 { half::bf16::from_f64(v) }
}

// ─── RealScalar for floats ────────────────────────────────────────────────────

impl RealScalar for f32 {
    fn min_value() -> f32  { f32::MIN }
    fn max_value() -> f32  { f32::MAX }
    fn abs(self) -> f32    { f32::abs(self) }
}

impl RealScalar for f64 {
    fn min_value() -> f64  { f64::MIN }
    fn max_value() -> f64  { f64::MAX }
    fn abs(self) -> f64    { f64::abs(self) }
}

impl RealScalar for half::f16 {
    fn min_value() -> Self { half::f16::MIN }
    fn max_value() -> Self { half::f16::MAX }
    fn abs(self)   -> Self { half::f16::from_f32(self.to_f32().abs()) }
}

impl RealScalar for half::bf16 {
    fn min_value() -> Self { half::bf16::MIN }
    fn max_value() -> Self { half::bf16::MAX }
    fn abs(self)   -> Self { half::bf16::from_f32(self.to_f32().abs()) }
}

// ─── FloatScalar for f32 / f64 ───────────────────────────────────────────────

macro_rules! impl_float_scalar_native {
    ($ty:ty) => {
        impl FloatScalar for $ty {
            fn nan()          -> $ty { <$ty>::NAN }
            fn infinity()     -> $ty { <$ty>::INFINITY }
            fn neg_infinity() -> $ty { <$ty>::NEG_INFINITY }

            #[inline] fn is_nan(self)           -> bool { <$ty>::is_nan(self) }
            #[inline] fn is_infinite(self)      -> bool { <$ty>::is_infinite(self) }
            #[inline] fn is_finite(self)        -> bool { <$ty>::is_finite(self) }
            #[inline] fn is_sign_positive(self) -> bool { <$ty>::is_sign_positive(self) }
            #[inline] fn is_sign_negative(self) -> bool { <$ty>::is_sign_negative(self) }

            #[inline] fn sqrt(self)   -> $ty { <$ty>::sqrt(self) }
            #[inline] fn ln(self)     -> $ty { <$ty>::ln(self) }
            #[inline] fn log2(self)   -> $ty { <$ty>::log2(self) }
            #[inline] fn log10(self)  -> $ty { <$ty>::log10(self) }
            #[inline] fn exp(self)    -> $ty { <$ty>::exp(self) }
            #[inline] fn exp2(self)   -> $ty { <$ty>::exp2(self) }
            #[inline] fn powi(self, n: i32) -> $ty { <$ty>::powi(self, n) }
            #[inline] fn powf(self, n: $ty) -> $ty { <$ty>::powf(self, n) }
            #[inline] fn floor(self)  -> $ty { <$ty>::floor(self) }
            #[inline] fn ceil(self)   -> $ty { <$ty>::ceil(self) }
            #[inline] fn round(self)  -> $ty { <$ty>::round(self) }
            #[inline] fn trunc(self)  -> $ty { <$ty>::trunc(self) }
            #[inline] fn fract(self)  -> $ty { <$ty>::fract(self) }
            #[inline] fn mul_add(self, a: $ty, b: $ty) -> $ty { <$ty>::mul_add(self, a, b) }

            fn epsilon()      -> $ty { <$ty>::EPSILON }
            fn min_positive() -> $ty { <$ty>::MIN_POSITIVE }

            #[inline] fn to_f32(self) -> f32 { self as f32 }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
        }
    };
}

impl_float_scalar_native!(f32);
impl_float_scalar_native!(f64);

// ─── FloatScalar for half-precision (via f32 roundtrip) ──────────────────────

macro_rules! impl_float_scalar_half {
    ($ty:ty, $from_f32:expr, $from_f64:expr) => {
        impl FloatScalar for $ty {
            fn nan()          -> $ty { <$ty>::NAN }
            fn infinity()     -> $ty { <$ty>::INFINITY }
            fn neg_infinity() -> $ty { <$ty>::NEG_INFINITY }

            #[inline] fn is_nan(self)           -> bool { <$ty>::is_nan(self) }
            #[inline] fn is_infinite(self)      -> bool { <$ty>::is_infinite(self) }
            #[inline] fn is_finite(self)        -> bool { <$ty>::is_finite(self) }
            #[inline] fn is_sign_positive(self) -> bool { self.to_f32() >= 0.0 }
            #[inline] fn is_sign_negative(self) -> bool { self.to_f32() <  0.0 }

            // All ops go via f32 — sufficient for half-precision accuracy.
            #[inline] fn sqrt(self)   -> $ty { $from_f32(self.to_f32().sqrt()) }
            #[inline] fn ln(self)     -> $ty { $from_f32(self.to_f32().ln()) }
            #[inline] fn log2(self)   -> $ty { $from_f32(self.to_f32().log2()) }
            #[inline] fn log10(self)  -> $ty { $from_f32(self.to_f32().log10()) }
            #[inline] fn exp(self)    -> $ty { $from_f32(self.to_f32().exp()) }
            #[inline] fn exp2(self)   -> $ty { $from_f32(self.to_f32().exp2()) }
            #[inline] fn powi(self, n: i32) -> $ty { $from_f32(self.to_f32().powi(n)) }
            #[inline] fn powf(self, n: $ty) -> $ty { $from_f32(self.to_f32().powf(n.to_f32())) }
            #[inline] fn floor(self)  -> $ty { $from_f32(self.to_f32().floor()) }
            #[inline] fn ceil(self)   -> $ty { $from_f32(self.to_f32().ceil()) }
            #[inline] fn round(self)  -> $ty { $from_f32(self.to_f32().round()) }
            #[inline] fn trunc(self)  -> $ty { $from_f32(self.to_f32().trunc()) }
            #[inline] fn fract(self)  -> $ty { $from_f32(self.to_f32().fract()) }
            #[inline] fn mul_add(self, a: $ty, b: $ty) -> $ty {
                $from_f32(self.to_f32().mul_add(a.to_f32(), b.to_f32()))
            }
            fn epsilon()      -> $ty { <$ty>::EPSILON }
            fn min_positive() -> $ty { <$ty>::MIN_POSITIVE }

            #[inline] fn to_f32(self) -> f32 { <$ty>::to_f32(self) }
            #[inline] fn to_f64(self) -> f64 { <$ty>::to_f64(self) }
        }
    };
}

impl_float_scalar_half!(half::f16,  half::f16::from_f32,  half::f16::from_f64);
impl_float_scalar_half!(half::bf16, half::bf16::from_f32, half::bf16::from_f64);

// ─── Scalar for Complex types ─────────────────────────────────────────────────

impl Scalar for Complex<f32> {
    const DTYPE:    DType       = DType::C64;
    const ZERO:     Complex<f32> = Complex { re: 0.0, im: 0.0 };
    const ONE:      Complex<f32> = Complex { re: 1.0, im: 0.0 };
    const ITEMSIZE: usize       = 8;

    #[inline] fn to_f64_lossy(self) -> f64          { self.norm() as f64 }
    #[inline] fn from_f64_lossy(v: f64) -> Self { Complex { re: v as f32, im: 0.0 } }
}

impl Scalar for Complex<f64> {
    const DTYPE:    DType       = DType::C128;
    const ZERO:     Complex<f64> = Complex { re: 0.0, im: 0.0 };
    const ONE:      Complex<f64> = Complex { re: 1.0, im: 0.0 };
    const ITEMSIZE: usize       = 16;

    #[inline] fn to_f64_lossy(self) -> f64          { self.norm() }
    #[inline] fn from_f64_lossy(v: f64) -> Self { Complex { re: v, im: 0.0 } }
}

// ─── ComplexScalar impls ──────────────────────────────────────────────────────

macro_rules! impl_complex_scalar {
    ($ty:ty, $real:ty, $dtype:expr) => {
        impl ComplexScalar for Complex<$real> {
            type Real = $real;

            #[inline] fn from_re_im(re: $real, im: $real) -> Self { Complex { re, im } }
            #[inline] fn re(self) -> $real   { self.re }
            #[inline] fn im(self) -> $real   { self.im }
            #[inline] fn conj(self) -> Self  { Complex::conj(&self) }
            #[inline] fn norm(self) -> $real { Complex::norm(self) }
            #[inline] fn norm_sqr(self) -> $real { Complex::norm_sqr(&self) }
            #[inline] fn arg(self) -> $real  { Complex::arg(self) }
            #[inline] fn is_nan(self)      -> bool { self.re.is_nan() || self.im.is_nan() }
            #[inline] fn is_infinite(self) -> bool { self.re.is_infinite() || self.im.is_infinite() }
            #[inline] fn is_finite(self)   -> bool { self.re.is_finite() && self.im.is_finite() }
        }
    };
}

impl_complex_scalar!(Complex<f32>, f32, DType::C64);
impl_complex_scalar!(Complex<f64>, f64, DType::C128);

// ─── compile-time check: all 15 types implement Scalar ───────────────────────
// These are const-evaluated, so they fail at compile time if any impl is missing.

const _SCALAR_IMPLS: () = {
    const fn check<T: Scalar>() {}
    check::<bool>();
    check::<i8>();  check::<i16>(); check::<i32>(); check::<i64>();
    check::<u8>();  check::<u16>(); check::<u32>(); check::<u64>();
    check::<half::f16>();  check::<half::bf16>();
    check::<f32>(); check::<f64>();
    check::<Complex<f32>>(); check::<Complex<f64>>();
};
