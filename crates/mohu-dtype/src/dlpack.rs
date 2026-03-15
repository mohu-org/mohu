/// DLPack type-code mapping for `DType`.
///
/// DLPack represents data types with a `DLDataType` C struct:
///
/// ```c
/// typedef struct {
///     uint8_t  code;   // type kind
///     uint8_t  bits;   // bits per element (per lane)
///     uint16_t lanes;  // number of SIMD lanes (1 for scalar)
/// } DLDataType;
/// ```
///
/// Kind codes:
///
/// | `code` | Meaning      |
/// |--------|--------------|
/// | 0      | `kDLInt`     |
/// | 1      | `kDLUInt`    |
/// | 2      | `kDLFloat`   |
/// | 4      | `kDLBfloat`  |
/// | 5      | `kDLComplex` |
///
/// Bool is represented as `kDLUInt` with 8 bits (same as PyTorch/JAX).
use mohu_error::{MohuError, MohuResult};

use crate::dtype::DType;

// ─── DLDataTypeCode ─────────────────────────────────────────────────────────

/// The `code` field of a DLPack `DLDataType`.
///
/// These constants match the DLPack specification v0.8.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DLDataTypeCode {
    Int     = 0,
    UInt    = 1,
    Float   = 2,
    BFloat  = 4,
    Complex = 5,
}

impl DLDataTypeCode {
    /// Constructs a `DLDataTypeCode` from its raw `u8` value.
    pub fn from_u8(code: u8) -> MohuResult<Self> {
        match code {
            0 => Ok(Self::Int),
            1 => Ok(Self::UInt),
            2 => Ok(Self::Float),
            4 => Ok(Self::BFloat),
            5 => Ok(Self::Complex),
            n => Err(MohuError::DLPackInvalid(format!(
                "unknown DLDataType code {n} — expected 0 (int), 1 (uint), \
                 2 (float), 4 (bfloat), or 5 (complex)"
            ))),
        }
    }
}

// ─── DLDataType ──────────────────────────────────────────────────────────────

/// A parsed representation of a DLPack `DLDataType` struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLDataType {
    pub code:  DLDataTypeCode,
    /// Bits per scalar element (per lane).
    pub bits:  u8,
    /// Number of SIMD lanes.  Must be 1 for mohu arrays.
    pub lanes: u16,
}

impl DLDataType {
    /// Constructs a `DLDataType` with `lanes = 1`.
    pub const fn scalar(code: DLDataTypeCode, bits: u8) -> Self {
        Self { code, bits, lanes: 1 }
    }

    /// Parses from raw `(code, bits, lanes)` triple.
    pub fn from_raw(code: u8, bits: u8, lanes: u16) -> MohuResult<Self> {
        let code = DLDataTypeCode::from_u8(code)?;
        Ok(Self { code, bits, lanes })
    }

    /// Returns `(code as u8, bits, lanes)` for writing into a C struct.
    pub fn to_raw(self) -> (u8, u8, u16) {
        (self.code as u8, self.bits, self.lanes)
    }
}

// ─── DType ↔ DLDataType ──────────────────────────────────────────────────────

impl DType {
    /// Converts this `DType` to a DLPack `DLDataType`.
    ///
    /// All mohu types use `lanes = 1`.
    ///
    /// ```rust
    /// # use mohu_dtype::{dtype::DType, dlpack::DLDataTypeCode};
    /// let dt = DType::F32.to_dlpack();
    /// assert_eq!(dt.code, DLDataTypeCode::Float);
    /// assert_eq!(dt.bits, 32);
    /// assert_eq!(dt.lanes, 1);
    /// ```
    pub const fn to_dlpack(self) -> DLDataType {
        use DLDataTypeCode::*;
        match self {
            Self::Bool => DLDataType::scalar(UInt,    8),
            Self::I8   => DLDataType::scalar(Int,     8),
            Self::I16  => DLDataType::scalar(Int,    16),
            Self::I32  => DLDataType::scalar(Int,    32),
            Self::I64  => DLDataType::scalar(Int,    64),
            Self::U8   => DLDataType::scalar(UInt,    8),
            Self::U16  => DLDataType::scalar(UInt,   16),
            Self::U32  => DLDataType::scalar(UInt,   32),
            Self::U64  => DLDataType::scalar(UInt,   64),
            Self::F16  => DLDataType::scalar(Float,  16),
            Self::BF16 => DLDataType::scalar(BFloat, 16),
            Self::F32  => DLDataType::scalar(Float,  32),
            Self::F64  => DLDataType::scalar(Float,  64),
            Self::C64  => DLDataType::scalar(Complex, 64),
            Self::C128 => DLDataType::scalar(Complex,128),
        }
    }

    /// Constructs a `DType` from DLPack `(code, bits, lanes)`.
    ///
    /// Returns an error if:
    /// - `lanes != 1` (mohu does not support multi-lane/vector dtypes)
    /// - the `(code, bits)` combination has no mohu equivalent
    ///
    /// Note: Bool and U8 both map to `(kDLUInt, 8)` in DLPack.  Since there
    /// is no Bool kind code in DLPack, this function returns `U8` for that
    /// combination.  Callers that need to round-trip Bool should track the
    /// dtype separately.
    pub fn from_dlpack(code: u8, bits: u8, lanes: u16) -> MohuResult<Self> {
        if lanes != 1 {
            return Err(MohuError::DLPackInvalid(format!(
                "mohu does not support multi-lane dtypes (lanes={lanes}); \
                 only lanes=1 is supported"
            )));
        }
        let kind = DLDataTypeCode::from_u8(code)?;
        match (kind, bits) {
            (DLDataTypeCode::Int,     8)  => Ok(Self::I8),
            (DLDataTypeCode::Int,    16)  => Ok(Self::I16),
            (DLDataTypeCode::Int,    32)  => Ok(Self::I32),
            (DLDataTypeCode::Int,    64)  => Ok(Self::I64),
            (DLDataTypeCode::UInt,   8)   => Ok(Self::U8),   // Bool also maps here
            (DLDataTypeCode::UInt,  16)   => Ok(Self::U16),
            (DLDataTypeCode::UInt,  32)   => Ok(Self::U32),
            (DLDataTypeCode::UInt,  64)   => Ok(Self::U64),
            (DLDataTypeCode::Float, 16)   => Ok(Self::F16),
            (DLDataTypeCode::Float, 32)   => Ok(Self::F32),
            (DLDataTypeCode::Float, 64)   => Ok(Self::F64),
            (DLDataTypeCode::BFloat, 16)  => Ok(Self::BF16),
            (DLDataTypeCode::Complex, 64) => Ok(Self::C64),
            (DLDataTypeCode::Complex,128) => Ok(Self::C128),
            (_, b) => Err(MohuError::DLPackUnsupportedDType {
                code,
                bits: b,
                lanes,
            }),
        }
    }
}

// ─── DLPack supported device check ───────────────────────────────────────────

/// DLPack device type codes.
///
/// Only `Cpu` (1) is supported by mohu.  GPU device types are listed here
/// for use in error messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DLDeviceType {
    Cpu        = 1,
    Cuda       = 2,
    CpuPinned  = 3,
    OpenCL     = 4,
    Vulkan     = 7,
    Metal      = 8,
    Vpi        = 9,
    Rocm       = 10,
    ExtDev     = 12,
    CudaManaged= 13,
    OneApi     = 14,
    WebGpu     = 15,
    Hexagon    = 16,
}

impl DLDeviceType {
    /// Parses a device type from its raw `i32` code.
    pub fn from_i32(code: i32) -> Option<Self> {
        match code {
            1  => Some(Self::Cpu),
            2  => Some(Self::Cuda),
            3  => Some(Self::CpuPinned),
            4  => Some(Self::OpenCL),
            7  => Some(Self::Vulkan),
            8  => Some(Self::Metal),
            9  => Some(Self::Vpi),
            10 => Some(Self::Rocm),
            12 => Some(Self::ExtDev),
            13 => Some(Self::CudaManaged),
            14 => Some(Self::OneApi),
            15 => Some(Self::WebGpu),
            16 => Some(Self::Hexagon),
            _  => None,
        }
    }

    /// Returns `true` if this device type is CPU-resident.
    ///
    /// mohu supports `Cpu` and `CpuPinned` (which is still CPU-accessible).
    pub fn is_cpu_resident(self) -> bool {
        matches!(self, Self::Cpu | Self::CpuPinned)
    }

    /// Human-readable device name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu         => "CPU",
            Self::Cuda        => "CUDA",
            Self::CpuPinned   => "CPU (pinned)",
            Self::OpenCL      => "OpenCL",
            Self::Vulkan      => "Vulkan",
            Self::Metal       => "Metal",
            Self::Vpi         => "VPI",
            Self::Rocm        => "ROCm",
            Self::ExtDev      => "ExtDev",
            Self::CudaManaged => "CUDA (managed)",
            Self::OneApi      => "OneAPI",
            Self::WebGpu      => "WebGPU",
            Self::Hexagon     => "Hexagon",
        }
    }
}

/// Validates that a DLPack device code is CPU-resident.
///
/// Returns `Err(DLPackUnsupportedDevice)` for GPU devices.
pub fn assert_cpu_device(device_type: i32) -> MohuResult<()> {
    match DLDeviceType::from_i32(device_type) {
        Some(d) if d.is_cpu_resident() => Ok(()),
        _ => Err(MohuError::DLPackUnsupportedDevice { device_type }),
    }
}
