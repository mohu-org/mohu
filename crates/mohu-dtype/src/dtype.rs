use std::fmt;
use mohu_error::{MohuError, MohuResult};

/// The runtime type tag for every element type mohu supports.
///
/// A `DType` value is the runtime representation of a type — it can be
/// stored, passed as a function argument, read from a file header, or
/// received from a Python caller.  The compile-time counterpart is the
/// [`Scalar`](crate::scalar::Scalar) trait.
///
/// # Variant ordering
///
/// Variants are assigned stable `repr(u8)` codes used in the DLPack
/// integration, the promotion table, and serialised formats.
/// **Never reorder or renumber existing variants.**
///
/// # Size table
///
/// | Variant | Rust type          | Bytes | Alignment |
/// |---------|--------------------|-------|-----------|
/// | Bool    | `bool`             |   1   |     1     |
/// | I8      | `i8`               |   1   |     1     |
/// | I16     | `i16`              |   2   |     2     |
/// | I32     | `i32`              |   4   |     4     |
/// | I64     | `i64`              |   8   |     8     |
/// | U8      | `u8`               |   1   |     1     |
/// | U16     | `u16`              |   2   |     2     |
/// | U32     | `u32`              |   4   |     4     |
/// | U64     | `u64`              |   8   |     8     |
/// | F16     | `half::f16`        |   2   |     2     |
/// | BF16    | `half::bf16`       |   2   |     2     |
/// | F32     | `f32`              |   4   |     4     |
/// | F64     | `f64`              |   8   |     8     |
/// | C64     | `Complex<f32>`     |   8   |     4     |
/// | C128    | `Complex<f64>`     |  16   |     8     |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    I8   = 1,
    I16  = 2,
    I32  = 3,
    I64  = 4,
    U8   = 5,
    U16  = 6,
    U32  = 7,
    U64  = 8,
    F16  = 9,
    BF16 = 10,
    F32  = 11,
    F64  = 12,
    C64  = 13,
    C128 = 14,
}

/// Total number of `DType` variants — used to size static tables.
pub const DTYPE_COUNT: usize = 15;

/// All `DType` variants in definition order.
pub const ALL_DTYPES: [DType; DTYPE_COUNT] = [
    DType::Bool,
    DType::I8,  DType::I16,  DType::I32,  DType::I64,
    DType::U8,  DType::U16,  DType::U32,  DType::U64,
    DType::F16, DType::BF16, DType::F32,  DType::F64,
    DType::C64, DType::C128,
];

impl DType {
    // -------------------------------------------------------------------------
    // Memory layout
    // -------------------------------------------------------------------------

    /// Size of one element in bytes.
    ///
    /// ```
    /// # use mohu_dtype::dtype::DType;
    /// assert_eq!(DType::F64.itemsize(), 8);
    /// assert_eq!(DType::C128.itemsize(), 16);
    /// assert_eq!(DType::Bool.itemsize(), 1);
    /// ```
    pub const fn itemsize(self) -> usize {
        match self {
            Self::Bool | Self::I8 | Self::U8                      => 1,
            Self::I16  | Self::U16 | Self::F16 | Self::BF16       => 2,
            Self::I32  | Self::U32 | Self::F32                    => 4,
            Self::I64  | Self::U64 | Self::F64 | Self::C64        => 8,
            Self::C128                                             => 16,
        }
    }

    /// Required pointer alignment in bytes.  Matches the alignment of the
    /// corresponding Rust primitive.
    pub const fn alignment(self) -> usize {
        match self {
            Self::Bool | Self::I8  | Self::U8                     => 1,
            Self::I16  | Self::U16 | Self::F16  | Self::BF16      => 2,
            Self::I32  | Self::U32 | Self::F32  | Self::C64       => 4,
            Self::I64  | Self::U64 | Self::F64  | Self::C128      => 8,
        }
    }

    /// Width of one element in bits.
    pub const fn bit_width(self) -> u32 {
        (self.itemsize() * 8) as u32
    }

    // -------------------------------------------------------------------------
    // Classification predicates
    // -------------------------------------------------------------------------

    /// Returns `true` if this is `Bool`.
    #[inline] pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns `true` if this is any integer type (signed or unsigned).
    #[inline] pub const fn is_integer(self) -> bool {
        matches!(self, Self::I8|Self::I16|Self::I32|Self::I64
                     | Self::U8|Self::U16|Self::U32|Self::U64)
    }

    /// Returns `true` if this is a signed integer.
    #[inline] pub const fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8|Self::I16|Self::I32|Self::I64)
    }

    /// Returns `true` if this is an unsigned integer.
    #[inline] pub const fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8|Self::U16|Self::U32|Self::U64)
    }

    /// Returns `true` if this is a non-complex floating-point type
    /// (F16, BF16, F32, or F64).
    #[inline] pub const fn is_float(self) -> bool {
        matches!(self, Self::F16|Self::BF16|Self::F32|Self::F64)
    }

    /// Returns `true` if this is a complex type (C64 or C128).
    #[inline] pub const fn is_complex(self) -> bool {
        matches!(self, Self::C64|Self::C128)
    }

    /// Returns `true` for any floating-point type, including complex.
    #[inline] pub const fn is_floating_point(self) -> bool {
        self.is_float() || self.is_complex()
    }

    /// Returns `true` for any type that supports arithmetic (everything
    /// except `Bool`).
    #[inline] pub const fn is_numeric(self) -> bool {
        !matches!(self, Self::Bool)
    }

    /// Returns `true` for types that have a total order (integers and
    /// real floats — not complex, not bool).
    #[inline] pub const fn is_ordered(self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Returns `true` for standard IEEE 754 precision (F32 or F64).
    #[inline] pub const fn is_standard_float(self) -> bool {
        matches!(self, Self::F32|Self::F64)
    }

    /// Returns `true` for `BF16` (brain float).
    #[inline] pub const fn is_brain_float(self) -> bool {
        matches!(self, Self::BF16)
    }

    // -------------------------------------------------------------------------
    // Type conversion helpers
    // -------------------------------------------------------------------------

    /// For complex types, returns the dtype of the real / imaginary component.
    /// C64 → F32, C128 → F64.  For all other types, returns `self`.
    pub const fn real_dtype(self) -> DType {
        match self {
            Self::C64  => Self::F32,
            Self::C128 => Self::F64,
            other => other,
        }
    }

    /// Converts a real float dtype to the corresponding complex dtype.
    /// F32 → C64, F64 → C128.  All other types return `self`.
    pub const fn complex_dtype(self) -> DType {
        match self {
            Self::F32 => Self::C64,
            Self::F64 => Self::C128,
            other => other,
        }
    }

    /// Converts unsigned integer to same-width signed integer.
    /// U8 → I8, U16 → I16, U32 → I32, U64 → I64.  Others return `self`.
    pub const fn as_signed(self) -> DType {
        match self {
            Self::U8  => Self::I8,
            Self::U16 => Self::I16,
            Self::U32 => Self::I32,
            Self::U64 => Self::I64,
            other => other,
        }
    }

    /// Converts signed integer to same-width unsigned integer.
    /// I8 → U8, I16 → U16, I32 → U32, I64 → U64.  Others return `self`.
    pub const fn as_unsigned(self) -> DType {
        match self {
            Self::I8  => Self::U8,
            Self::I16 => Self::U16,
            Self::I32 => Self::U32,
            Self::I64 => Self::U64,
            other => other,
        }
    }

    /// Widens this dtype by one step:
    /// F32 → F64, I32 → I64, U8 → U16, Bool → U8, F16/BF16 → F32, C64 → C128.
    /// Types already at maximum width return `self`.
    pub const fn widen(self) -> DType {
        match self {
            Self::Bool => Self::U8,
            Self::I8   => Self::I16,
            Self::I16  => Self::I32,
            Self::I32  => Self::I64,
            Self::I64  => Self::I64,
            Self::U8   => Self::U16,
            Self::U16  => Self::U32,
            Self::U32  => Self::U64,
            Self::U64  => Self::U64,
            Self::F16  => Self::F32,
            Self::BF16 => Self::F32,
            Self::F32  => Self::F64,
            Self::F64  => Self::F64,
            Self::C64  => Self::C128,
            Self::C128 => Self::C128,
        }
    }

    /// Narrows this dtype by one step.  Returns `None` for minimum-width
    /// types (Bool, I8, U8, F16, BF16).
    pub const fn narrow(self) -> Option<DType> {
        match self {
            Self::I16  => Some(Self::I8),
            Self::I32  => Some(Self::I16),
            Self::I64  => Some(Self::I32),
            Self::U16  => Some(Self::U8),
            Self::U32  => Some(Self::U16),
            Self::U64  => Some(Self::U32),
            Self::F32  => Some(Self::F16),
            Self::F64  => Some(Self::F32),
            Self::C128 => Some(Self::C64),
            _ => None,
        }
    }

    /// Returns the smallest floating-point dtype that can represent all
    /// values of this integer dtype without precision loss.
    ///
    /// I8/U8 → F16, I16/U16 → F32, I32/U32/I64/U64 → F64.
    /// For float and complex, returns `self`. For Bool, returns F16.
    pub const fn to_float(self) -> DType {
        match self {
            Self::Bool | Self::I8  | Self::U8  => Self::F16,
            Self::I16  | Self::U16             => Self::F32,
            Self::I32  | Self::U32
                | Self::I64 | Self::U64        => Self::F64,
            other => other,
        }
    }

    // -------------------------------------------------------------------------
    // String representation  (NumPy-compatible)
    // -------------------------------------------------------------------------

    /// Returns the canonical NumPy dtype name, e.g. `"float32"`, `"int64"`.
    pub const fn numpy_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8   => "int8",
            Self::I16  => "int16",
            Self::I32  => "int32",
            Self::I64  => "int64",
            Self::U8   => "uint8",
            Self::U16  => "uint16",
            Self::U32  => "uint32",
            Self::U64  => "uint64",
            Self::F16  => "float16",
            Self::BF16 => "bfloat16",
            Self::F32  => "float32",
            Self::F64  => "float64",
            Self::C64  => "complex64",
            Self::C128 => "complex128",
        }
    }

    /// Returns the single-character NumPy type-code.  BF16 has no standard
    /// NumPy char and returns `None`.
    ///
    /// | DType | Code |   | DType | Code |
    /// |-------|------|---|-------|------|
    /// | Bool  | `?`  |   | U8    | `B`  |
    /// | I8    | `b`  |   | U16   | `H`  |
    /// | I16   | `h`  |   | U32   | `I`  |
    /// | I32   | `i`  |   | U64   | `L`  |
    /// | I64   | `l`  |   | F16   | `e`  |
    /// | F32   | `f`  |   | F64   | `d`  |
    /// | C64   | `F`  |   | C128  | `D`  |
    pub const fn numpy_char(self) -> Option<char> {
        match self {
            Self::Bool => Some('?'),
            Self::I8   => Some('b'),
            Self::I16  => Some('h'),
            Self::I32  => Some('i'),
            Self::I64  => Some('l'),
            Self::U8   => Some('B'),
            Self::U16  => Some('H'),
            Self::U32  => Some('I'),
            Self::U64  => Some('L'),
            Self::F16  => Some('e'),
            Self::BF16 => None,
            Self::F32  => Some('f'),
            Self::F64  => Some('d'),
            Self::C64  => Some('F'),
            Self::C128 => Some('D'),
        }
    }

    /// Returns the broad NumPy array-interface kind character:
    /// `'b'` bool, `'i'` signed int, `'u'` unsigned int,
    /// `'f'` float, `'c'` complex.
    pub const fn kind_char(self) -> char {
        match self {
            Self::Bool => 'b',
            Self::I8|Self::I16|Self::I32|Self::I64         => 'i',
            Self::U8|Self::U16|Self::U32|Self::U64         => 'u',
            Self::F16|Self::BF16|Self::F32|Self::F64       => 'f',
            Self::C64|Self::C128                           => 'c',
        }
    }

    /// Returns the array-interface type string, e.g. `"<f4"` (little-endian
    /// F32), `"|b1"` (Bool).
    pub fn array_interface_typestr(self) -> String {
        let endian = if self.itemsize() == 1 { '|' } else { '<' };
        format!("{}{}{}", endian, self.kind_char(), self.itemsize())
    }

    // -------------------------------------------------------------------------
    // Numeric value bounds
    // -------------------------------------------------------------------------

    /// Minimum finite value as `f64`, or `None` for complex types.
    pub fn min_as_f64(self) -> Option<f64> {
        match self {
            Self::Bool => Some(0.0),
            Self::I8   => Some(i8::MIN as f64),
            Self::I16  => Some(i16::MIN as f64),
            Self::I32  => Some(i32::MIN as f64),
            Self::I64  => Some(i64::MIN as f64),
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => Some(0.0),
            Self::F16  => Some(half::f16::MIN.to_f64()),
            Self::BF16 => Some(half::bf16::MIN.to_f64()),
            Self::F32  => Some(f32::MIN as f64),
            Self::F64  => Some(f64::MIN),
            Self::C64 | Self::C128 => None,
        }
    }

    /// Maximum finite value as `f64`, or `None` for complex types.
    pub fn max_as_f64(self) -> Option<f64> {
        match self {
            Self::Bool => Some(1.0),
            Self::I8   => Some(i8::MAX as f64),
            Self::I16  => Some(i16::MAX as f64),
            Self::I32  => Some(i32::MAX as f64),
            Self::I64  => Some(i64::MAX as f64),
            Self::U8   => Some(u8::MAX as f64),
            Self::U16  => Some(u16::MAX as f64),
            Self::U32  => Some(u32::MAX as f64),
            Self::U64  => Some(u64::MAX as f64),
            Self::F16  => Some(half::f16::MAX.to_f64()),
            Self::BF16 => Some(half::bf16::MAX.to_f64()),
            Self::F32  => Some(f32::MAX as f64),
            Self::F64  => Some(f64::MAX),
            Self::C64 | Self::C128 => None,
        }
    }

    /// Machine epsilon for floating-point types, or `None` for all others.
    pub fn epsilon_as_f64(self) -> Option<f64> {
        match self {
            Self::F16  => Some(half::f16::EPSILON.to_f64()),
            Self::BF16 => Some(half::bf16::EPSILON.to_f64()),
            Self::F32  => Some(f32::EPSILON as f64),
            Self::F64  => Some(f64::EPSILON),
            _ => None,
        }
    }

    /// Smallest positive normalised value for float types, or `None`.
    pub fn min_positive_as_f64(self) -> Option<f64> {
        match self {
            Self::F16  => Some(half::f16::MIN_POSITIVE.to_f64()),
            Self::BF16 => Some(half::bf16::MIN_POSITIVE.to_f64()),
            Self::F32  => Some(f32::MIN_POSITIVE as f64),
            Self::F64  => Some(f64::MIN_POSITIVE),
            _ => None,
        }
    }

    /// Number of significant decimal digits for integer types, or `None`.
    pub const fn max_decimal_digits(self) -> Option<u32> {
        match self {
            Self::Bool             => Some(1),
            Self::I8  | Self::U8  => Some(3),
            Self::I16 | Self::U16 => Some(5),
            Self::I32 | Self::U32 => Some(10),
            Self::I64 | Self::U64 => Some(20),
            _ => None,
        }
    }

    // -------------------------------------------------------------------------
    // Stable numeric code
    // -------------------------------------------------------------------------

    /// Returns the stable `u8` discriminant code for this dtype.
    pub const fn as_u8(self) -> u8 { self as u8 }

    /// Constructs a `DType` from its `u8` code.
    pub fn from_u8(code: u8) -> MohuResult<Self> {
        match code {
            0  => Ok(Self::Bool),
            1  => Ok(Self::I8),
            2  => Ok(Self::I16),
            3  => Ok(Self::I32),
            4  => Ok(Self::I64),
            5  => Ok(Self::U8),
            6  => Ok(Self::U16),
            7  => Ok(Self::U32),
            8  => Ok(Self::U64),
            9  => Ok(Self::F16),
            10 => Ok(Self::BF16),
            11 => Ok(Self::F32),
            12 => Ok(Self::F64),
            13 => Ok(Self::C64),
            14 => Ok(Self::C128),
            n  => Err(MohuError::UnknownDType(format!("dtype code {n}"))),
        }
    }

    // -------------------------------------------------------------------------
    // Parsing
    // -------------------------------------------------------------------------

    /// Parses a dtype from a NumPy-compatible string.
    ///
    /// Accepts canonical names, single-char codes, shorthand with byte counts,
    /// and common aliases. Case-insensitive except for uppercase char codes
    /// (B, H, I, L, F, D).
    pub fn from_str(s: &str) -> MohuResult<Self> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "bool" | "bool_" | "?" | "b1"
                => Ok(Self::Bool),
            "int8"  | "i1" | "b" | "byte"
                => Ok(Self::I8),
            "int16" | "i2" | "h" | "short"
                => Ok(Self::I16),
            "int32" | "i4" | "i" | "int" | "int_"
                => Ok(Self::I32),
            "int64" | "i8" | "l" | "long" | "longlong"
                => Ok(Self::I64),
            "uint8"  | "u1" | "ubyte"
                => Ok(Self::U8),
            "uint16" | "u2" | "ushort"
                => Ok(Self::U16),
            "uint32" | "u4" | "uint"
                => Ok(Self::U32),
            "uint64" | "u8" | "ulong"
                => Ok(Self::U64),
            "float16" | "f2" | "half" | "f16"
                => Ok(Self::F16),
            "bfloat16" | "bf16"
                => Ok(Self::BF16),
            "float32" | "f4" | "float" | "single" | "f32"
                => Ok(Self::F32),
            "float64" | "f8" | "double" | "f64"
                => Ok(Self::F64),
            "complex64"  | "c8"  | "csingle" | "c64"
                => Ok(Self::C64),
            "complex128" | "c16" | "cdouble" | "c128"
                => Ok(Self::C128),
            _ => {
                // Uppercase single-char codes survive the lowercase pass
                // only if the original string was uppercase.
                match s.trim() {
                    "B" => Ok(Self::U8),
                    "H" => Ok(Self::U16),
                    "I" => Ok(Self::U32),
                    "L" => Ok(Self::U64),
                    "F" => Ok(Self::C64),
                    "D" => Ok(Self::C128),
                    other => Err(MohuError::UnknownDType(other.to_string())),
                }
            }
        }
    }

    /// Returns an iterator over all variants in definition order.
    pub fn all() -> impl Iterator<Item = DType> {
        ALL_DTYPES.iter().copied()
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.numpy_str())
    }
}

// ─── TryFrom ─────────────────────────────────────────────────────────────────

impl TryFrom<&str> for DType {
    type Error = MohuError;
    fn try_from(s: &str) -> MohuResult<Self> { DType::from_str(s) }
}

impl TryFrom<String> for DType {
    type Error = MohuError;
    fn try_from(s: String) -> MohuResult<Self> { DType::from_str(&s) }
}

// ─── serde (feature-gated) ───────────────────────────────────────────────────

#[cfg(feature = "serde")]
impl serde::Serialize for DType {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.numpy_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        DType::from_str(&s).map_err(serde::de::Error::custom)
    }
}

// ─── compile-time invariant checks ───────────────────────────────────────────

const _: () = {
    assert!(ALL_DTYPES.len() == DTYPE_COUNT);
    assert!(DType::Bool as u8 == 0);
    assert!(DType::C128 as u8 == 14);
};
