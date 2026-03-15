/// Integer type metadata — the Rust equivalent of `numpy.iinfo`.
///
/// Every field mirrors NumPy's `iinfo` object exactly.
///
/// # Example
///
/// ```rust
/// # use mohu_dtype::{dtype::DType, iinfo::IntInfo};
/// let info = IntInfo::of(DType::I32).unwrap();
/// assert_eq!(info.min, i32::MIN as i128);
/// assert_eq!(info.max, i32::MAX as u128);
/// assert_eq!(info.bits, 32);
/// assert!(info.is_signed);
/// ```
use mohu_error::{MohuError, MohuResult};

use crate::dtype::DType;

/// Machine-range metadata for an integer `DType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntInfo {
    /// The `DType` this info describes.
    pub dtype: DType,

    /// Total number of bits.
    pub bits: u32,

    /// Whether this is a signed type.
    pub is_signed: bool,

    /// Minimum value.  For unsigned types this is always 0.
    pub min: i128,

    /// Maximum value, as an unsigned 128-bit integer.
    /// Use `max as i128` for signed types when you know it fits.
    pub max: u128,
}

impl IntInfo {
    /// Returns `IntInfo` for the given dtype, or an error if the dtype is
    /// not an integer type.
    pub fn of(dtype: DType) -> MohuResult<Self> {
        match dtype {
            DType::I8   => Ok(Self::i8()),
            DType::I16  => Ok(Self::i16()),
            DType::I32  => Ok(Self::i32()),
            DType::I64  => Ok(Self::i64()),
            DType::U8   => Ok(Self::u8()),
            DType::U16  => Ok(Self::u16()),
            DType::U32  => Ok(Self::u32()),
            DType::U64  => Ok(Self::u64()),
            other => Err(MohuError::UnsupportedDType {
                op: "iinfo",
                dtype: other.to_string(),
            }),
        }
    }

    // ─── signed ────────────────────────────────────────────────────────────────

    pub const fn i8() -> Self {
        Self { dtype: DType::I8,  bits: 8,  is_signed: true,
               min: i8::MIN as i128,  max: i8::MAX as u128 }
    }
    pub const fn i16() -> Self {
        Self { dtype: DType::I16, bits: 16, is_signed: true,
               min: i16::MIN as i128, max: i16::MAX as u128 }
    }
    pub const fn i32() -> Self {
        Self { dtype: DType::I32, bits: 32, is_signed: true,
               min: i32::MIN as i128, max: i32::MAX as u128 }
    }
    pub const fn i64() -> Self {
        Self { dtype: DType::I64, bits: 64, is_signed: true,
               min: i64::MIN as i128, max: i64::MAX as u128 }
    }

    // ─── unsigned ──────────────────────────────────────────────────────────────

    pub const fn u8() -> Self {
        Self { dtype: DType::U8,  bits: 8,  is_signed: false,
               min: 0, max: u8::MAX as u128 }
    }
    pub const fn u16() -> Self {
        Self { dtype: DType::U16, bits: 16, is_signed: false,
               min: 0, max: u16::MAX as u128 }
    }
    pub const fn u32() -> Self {
        Self { dtype: DType::U32, bits: 32, is_signed: false,
               min: 0, max: u32::MAX as u128 }
    }
    pub const fn u64() -> Self {
        Self { dtype: DType::U64, bits: 64, is_signed: false,
               min: 0, max: u64::MAX as u128 }
    }

    // ─── convenience ───────────────────────────────────────────────────────────

    /// Returns `true` if the integer value `v` fits in this dtype without
    /// overflow.
    pub fn can_hold_i128(self, v: i128) -> bool {
        v >= self.min && (v as u128) <= self.max
    }

    /// Returns `true` if the unsigned value `v` fits in this dtype.
    pub fn can_hold_u128(self, v: u128) -> bool {
        if self.is_signed {
            v <= self.max
        } else {
            v <= self.max
        }
    }

    /// Returns the minimum scalar type (smallest integer dtype) that can
    /// represent the given signed value without overflow.
    pub fn minimum_signed_type_for(v: i64) -> DType {
        if v >= i8::MIN as i64  && v <= i8::MAX as i64  { return DType::I8;  }
        if v >= i16::MIN as i64 && v <= i16::MAX as i64 { return DType::I16; }
        if v >= i32::MIN as i64 && v <= i32::MAX as i64 { return DType::I32; }
        DType::I64
    }

    /// Returns the minimum scalar type that can represent the given unsigned value.
    pub fn minimum_unsigned_type_for(v: u64) -> DType {
        if v <= u8::MAX as u64  { return DType::U8;  }
        if v <= u16::MAX as u64 { return DType::U16; }
        if v <= u32::MAX as u64 { return DType::U32; }
        DType::U64
    }
}

impl std::fmt::Display for IntInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IntInfo({dtype}: bits={bits}, min={min}, max={max}, signed={signed})",
            dtype  = self.dtype,
            bits   = self.bits,
            min    = self.min,
            max    = self.max,
            signed = self.is_signed,
        )
    }
}
