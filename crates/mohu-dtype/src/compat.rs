/// Compatibility mappings between `DType` and external type systems.
///
/// This module maps mohu's `DType` to and from the type representations used
/// by adjacent ecosystem libraries, so that zero-copy interop works correctly
/// without each consumer re-implementing the mapping.
///
/// # Covered systems
///
/// | System | Direction | Notes |
/// |--------|-----------|-------|
/// | Python `struct` module | both | single-char format codes |
/// | Python buffer protocol (PEP 3118) | both | format string with endian prefix |
/// | NumPy array interface | mohu→numpy | `"<f4"` typestr |
/// | Apache Arrow | both (feature = `"arrow"`) | `DataType` enum |
///
/// # Byte order
///
/// mohu always stores arrays in the host's native byte order.
/// Buffer-protocol and array-interface strings are prefixed with
/// `<` (little-endian) for multi-byte types and `|` (not applicable)
/// for single-byte types, matching NumPy's convention on x86/ARM hosts.
use mohu_error::{MohuError, MohuResult};

use crate::dtype::DType;

// ─── ByteOrder ───────────────────────────────────────────────────────────────

/// Byte order used in serialised format strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ByteOrder {
    /// Native endian (the host CPU's native order).
    Native,
    /// Little-endian (least significant byte first).
    Little,
    /// Big-endian (most significant byte first).
    Big,
    /// Not applicable (single-byte types).
    NotApplicable,
}

impl ByteOrder {
    /// Returns the struct-module prefix character for this byte order.
    pub const fn struct_prefix(self) -> char {
        match self {
            Self::Native      => '=',
            Self::Little      => '<',
            Self::Big         => '>',
            Self::NotApplicable => '|',
        }
    }

    /// Returns the byte order that should be used for a dtype of the given
    /// item size.  Single-byte types use `NotApplicable`; multi-byte types
    /// use `Native`.
    pub const fn for_itemsize(itemsize: usize) -> Self {
        if itemsize == 1 { Self::NotApplicable } else { Self::Native }
    }

    /// Returns `true` if the current host is little-endian.
    pub fn host_is_little_endian() -> bool {
        u16::from_ne_bytes([1, 0]) == 1
    }

    /// Returns the concrete byte order of the host.
    pub fn host() -> Self {
        if Self::host_is_little_endian() { Self::Little } else { Self::Big }
    }
}

// ─── Python struct module format strings ─────────────────────────────────────

impl DType {
    /// Returns the Python `struct` module format character for this dtype.
    ///
    /// These are the single-character codes used in `struct.pack`/`unpack`
    /// and `ctypes`.  The returned string does **not** include an endian prefix.
    ///
    /// BF16 has no `struct` module code — `None` is returned.
    ///
    /// | DType | Code |   | DType | Code |
    /// |-------|------|---|-------|------|
    /// | Bool  | `?`  |   | U8    | `B`  |
    /// | I8    | `b`  |   | U16   | `H`  |
    /// | I16   | `h`  |   | U32   | `I`  |
    /// | I32   | `i`  |   | U64   | `Q`  |
    /// | I64   | `q`  |   | F32   | `f`  |
    /// | F16   | `e`  |   | F64   | `d`  |
    pub const fn struct_format_char(self) -> Option<char> {
        match self {
            Self::Bool => Some('?'),
            Self::I8   => Some('b'),
            Self::I16  => Some('h'),
            Self::I32  => Some('i'),
            Self::I64  => Some('q'),
            Self::U8   => Some('B'),
            Self::U16  => Some('H'),
            Self::U32  => Some('I'),
            Self::U64  => Some('Q'),
            Self::F16  => Some('e'),
            Self::BF16 => None,         // no struct code
            Self::F32  => Some('f'),
            Self::F64  => Some('d'),
            // Complex: struct encodes as two consecutive floats
            Self::C64  => None,
            Self::C128 => None,
        }
    }

    /// Returns the full Python `struct` format string including a native-endian
    /// prefix, or an error for types that have no struct representation.
    ///
    /// ```rust
    /// # use mohu_dtype::dtype::DType;
    /// assert_eq!(DType::F32.to_struct_format().unwrap(), "=f");
    /// assert_eq!(DType::I64.to_struct_format().unwrap(), "=q");
    /// ```
    pub fn to_struct_format(self) -> MohuResult<String> {
        let ch = self.struct_format_char().ok_or_else(|| {
            MohuError::UnsupportedDType {
                op:    "struct format",
                dtype: self.to_string(),
            }
        })?;
        Ok(format!("={ch}"))
    }

    // ─── PEP 3118 / Python buffer protocol ────────────────────────────────────

    /// Returns the PEP 3118 buffer protocol format string for this dtype.
    ///
    /// This is the string placed in `Py_buffer.format` when implementing
    /// the buffer protocol.  It includes an endian prefix (`<` or `|`) and
    /// is therefore directly usable with `memoryview`.
    ///
    /// Complex types use the `Z` prefix (`Zf` = complex64, `Zd` = complex128).
    /// BF16 uses a non-standard extension `bfloat16` is transmitted as `e`
    /// with a comment — for strict PEP 3118, BF16 should be exposed as `H`
    /// (raw bytes) and documented.
    ///
    /// ```rust
    /// # use mohu_dtype::dtype::DType;
    /// assert_eq!(DType::F32.buffer_format(),  "<f");
    /// assert_eq!(DType::C64.buffer_format(),  "<Zf");
    /// assert_eq!(DType::Bool.buffer_format(), "|?");
    /// ```
    pub fn buffer_format(self) -> &'static str {
        match self {
            Self::Bool => "|?",
            Self::I8   => "|b",
            Self::U8   => "|B",
            Self::I16  => "<h",
            Self::U16  => "<H",
            Self::I32  => "<i",
            Self::U32  => "<I",
            Self::I64  => "<q",
            Self::U64  => "<Q",
            Self::F16  => "<e",
            Self::BF16 => "<e",  // non-standard; same size, different format
            Self::F32  => "<f",
            Self::F64  => "<d",
            Self::C64  => "<Zf",
            Self::C128 => "<Zd",
        }
    }

    /// Attempts to parse a dtype from a PEP 3118 buffer format string.
    ///
    /// Strips leading endian characters (`<`, `>`, `=`, `@`, `!`, `|`) and
    /// parses the remaining format code.
    pub fn from_buffer_format(fmt: &str) -> MohuResult<Self> {
        let stripped = fmt.trim_start_matches(['<', '>', '=', '@', '!', '|', ' ']);
        match stripped {
            "?" => Ok(Self::Bool),
            "b" => Ok(Self::I8),
            "B" => Ok(Self::U8),
            "h" => Ok(Self::I16),
            "H" => Ok(Self::U16),
            "i" | "l" => Ok(Self::I32),
            "I" | "L" => Ok(Self::U32),
            "q" | "n" => Ok(Self::I64),
            "Q" | "N" => Ok(Self::U64),
            "e" => Ok(Self::F16),
            "f" => Ok(Self::F32),
            "d" => Ok(Self::F64),
            "Zf" => Ok(Self::C64),
            "Zd" => Ok(Self::C128),
            other => Err(MohuError::PythonUnsupportedBufferFormat {
                format: other.to_string(),
            }),
        }
    }

    // ─── ctypes ───────────────────────────────────────────────────────────────

    /// Returns the Python `ctypes` type name for this dtype, or `None` if
    /// ctypes has no direct equivalent.
    ///
    /// This is useful for `numpy.ctypeslib.ndpointer` and for writing
    /// C FFI wrappers in Python.
    pub const fn ctypes_name(self) -> Option<&'static str> {
        match self {
            Self::Bool => Some("c_bool"),
            Self::I8   => Some("c_int8"),
            Self::I16  => Some("c_int16"),
            Self::I32  => Some("c_int32"),
            Self::I64  => Some("c_int64"),
            Self::U8   => Some("c_uint8"),
            Self::U16  => Some("c_uint16"),
            Self::U32  => Some("c_uint32"),
            Self::U64  => Some("c_uint64"),
            Self::F32  => Some("c_float"),
            Self::F64  => Some("c_double"),
            _ => None,   // F16, BF16, complex have no ctypes equivalent
        }
    }

    // ─── C type name ──────────────────────────────────────────────────────────

    /// Returns the C type name used in generated C/CUDA kernels.
    pub const fn c_type_name(self) -> &'static str {
        match self {
            Self::Bool => "_Bool",
            Self::I8   => "int8_t",
            Self::I16  => "int16_t",
            Self::I32  => "int32_t",
            Self::I64  => "int64_t",
            Self::U8   => "uint8_t",
            Self::U16  => "uint16_t",
            Self::U32  => "uint32_t",
            Self::U64  => "uint64_t",
            Self::F16  => "__fp16",
            Self::BF16 => "__bfloat16",
            Self::F32  => "float",
            Self::F64  => "double",
            Self::C64  => "float _Complex",
            Self::C128 => "double _Complex",
        }
    }

    // ─── Rust type name ───────────────────────────────────────────────────────

    /// Returns the canonical Rust type name for this dtype.
    pub const fn rust_type_name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::I8   => "i8",
            Self::I16  => "i16",
            Self::I32  => "i32",
            Self::I64  => "i64",
            Self::U8   => "u8",
            Self::U16  => "u16",
            Self::U32  => "u32",
            Self::U64  => "u64",
            Self::F16  => "half::f16",
            Self::BF16 => "half::bf16",
            Self::F32  => "f32",
            Self::F64  => "f64",
            Self::C64  => "num_complex::Complex<f32>",
            Self::C128 => "num_complex::Complex<f64>",
        }
    }
}

// ─── Arrow integration (feature-gated) ───────────────────────────────────────

#[cfg(feature = "arrow")]
pub mod arrow_compat {
    use arrow::datatypes::DataType as ArrowDataType;
    use mohu_error::{MohuError, MohuResult};

    use crate::dtype::DType;

    impl DType {
        /// Converts this `DType` to an Apache Arrow `DataType`.
        ///
        /// Returns `None` for BF16, which Arrow does not yet have a standard
        /// type for (some Arrow implementations use a custom extension type).
        pub fn to_arrow(self) -> Option<ArrowDataType> {
            match self {
                Self::Bool => Some(ArrowDataType::Boolean),
                Self::I8   => Some(ArrowDataType::Int8),
                Self::I16  => Some(ArrowDataType::Int16),
                Self::I32  => Some(ArrowDataType::Int32),
                Self::I64  => Some(ArrowDataType::Int64),
                Self::U8   => Some(ArrowDataType::UInt8),
                Self::U16  => Some(ArrowDataType::UInt16),
                Self::U32  => Some(ArrowDataType::UInt32),
                Self::U64  => Some(ArrowDataType::UInt64),
                Self::F16  => Some(ArrowDataType::Float16),
                Self::F32  => Some(ArrowDataType::Float32),
                Self::F64  => Some(ArrowDataType::Float64),
                // BF16 / complex have no standard Arrow equivalent
                Self::BF16 | Self::C64 | Self::C128 => None,
            }
        }

        /// Constructs a `DType` from an Apache Arrow `DataType`.
        ///
        /// Returns `Err(ArrowUnsupportedType)` for Arrow types that have
        /// no mohu equivalent (timestamps, dictionaries, lists, etc.).
        pub fn from_arrow(dt: &ArrowDataType) -> MohuResult<Self> {
            match dt {
                ArrowDataType::Boolean   => Ok(Self::Bool),
                ArrowDataType::Int8      => Ok(Self::I8),
                ArrowDataType::Int16     => Ok(Self::I16),
                ArrowDataType::Int32     => Ok(Self::I32),
                ArrowDataType::Int64     => Ok(Self::I64),
                ArrowDataType::UInt8     => Ok(Self::U8),
                ArrowDataType::UInt16    => Ok(Self::U16),
                ArrowDataType::UInt32    => Ok(Self::U32),
                ArrowDataType::UInt64    => Ok(Self::U64),
                ArrowDataType::Float16   => Ok(Self::F16),
                ArrowDataType::Float32   => Ok(Self::F32),
                ArrowDataType::Float64   => Ok(Self::F64),
                other => Err(MohuError::ArrowUnsupportedType {
                    arrow_type: format!("{other:?}"),
                }),
            }
        }
    }
}
