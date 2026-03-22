use thiserror::Error;

/// The central error type for the entire mohu library.
///
/// Every crate in the mohu workspace returns `MohuResult<T>` which is
/// `Result<T, MohuError>`. Variants are grouped by domain. The enum is
/// `#[non_exhaustive]` so that new variants can be added without breaking
/// downstream crates between minor versions.
///
/// # Error domains
///
/// | Range    | Domain          |
/// |----------|-----------------|
/// | 1000–    | Shape           |
/// | 2000–    | DType           |
/// | 3000–    | Index / slice   |
/// | 4000–    | Buffer / memory |
/// | 5000–    | Compute / math  |
/// | 6000–    | I/O             |
/// | 7000–    | DLPack          |
/// | 8000–    | Arrow           |
/// | 9000–    | Python / PyO3   |
/// | 10000–   | General         |
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MohuError {
    // -------------------------------------------------------------------------
    // Shape & dimension errors  (1xxx)
    // -------------------------------------------------------------------------

    /// Two arrays have incompatible shapes for an element-wise operation.
    #[error(
        "shape mismatch: expected {expected:?}, got {got:?}\n\
         hint: shapes must be identical for this operation, or broadcastable"
    )]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    /// Two shapes cannot be broadcast together under NumPy-style rules.
    #[error(
        "cannot broadcast shapes {lhs:?} and {rhs:?}\n\
         hint: trailing dimensions must be equal or one of them must be 1"
    )]
    BroadcastError { lhs: Vec<usize>, rhs: Vec<usize> },

    /// An operation received an array with the wrong number of dimensions.
    #[error(
        "dimension mismatch: expected a {expected}D array, got {got}D\n\
         hint: use reshape() or squeeze() to adjust the number of dimensions"
    )]
    DimensionMismatch { expected: usize, got: usize },

    /// An axis index is out of the valid range for the array's dimensionality.
    #[error(
        "axis {axis} is out of range for a {ndim}D array\n\
         hint: valid axes are {valid}"
    )]
    AxisOutOfRange { axis: i64, ndim: usize, valid: String },

    /// An operation that requires at least one dimension was called on a scalar.
    #[error(
        "cannot perform this operation on a zero-dimensional (scalar) array\n\
         hint: use at_least_1d() to promote a scalar to a 1D array"
    )]
    ScalarArray,

    /// A dimension has size zero, making the operation undefined.
    #[error("dimension {axis} has size 0 — operations on empty arrays are undefined")]
    ZeroSizedDimension { axis: usize },

    /// The total number of elements overflows `usize`.
    #[error(
        "shape overflow: the product of dimensions exceeds usize::MAX ({max})\n\
         hint: split the array into smaller chunks"
    )]
    ShapeOverflow { max: usize },

    /// A reshape cannot be performed because the element counts differ.
    #[error(
        "reshape is impossible: cannot reshape array with {src_len} elements \
         into shape {dst_shape:?} ({dst_len} elements)"
    )]
    ReshapeIncompatible {
        src_len: usize,
        dst_shape: Vec<usize>,
        dst_len: usize,
    },

    /// `stack` or `concatenate` was called with an empty sequence of arrays.
    #[error("stack requires at least one array, but an empty sequence was given")]
    EmptyStackSequence,

    /// Arrays passed to `stack`/`concatenate` have mismatched shapes on non-concat axes.
    #[error(
        "all arrays passed to stack/concatenate must have the same shape except \
         on the concatenation axis — mismatch at index {index}: \
         expected {expected:?}, got {got:?}"
    )]
    ConcatShapeMismatch {
        index: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    // -------------------------------------------------------------------------
    // DType errors  (2xxx)
    // -------------------------------------------------------------------------

    /// An operation received arrays with incompatible data types.
    #[error(
        "dtype mismatch: expected {expected}, got {got}\n\
         hint: use array.astype(\"{expected}\") to cast"
    )]
    DTypeMismatch { expected: String, got: String },

    /// A type cast between two dtypes is not valid or would lose data.
    #[error("cannot cast {from} to {to}: {reason}")]
    InvalidCast { from: String, to: String, reason: String },

    /// A value exceeds the representable range of the target dtype.
    #[error(
        "value overflows {dtype}: {detail}\n\
         hint: use a wider dtype or clamp values before casting"
    )]
    Overflow { dtype: String, detail: String },

    /// A value is too small to be represented by the target dtype.
    #[error(
        "value underflows {dtype}: {detail}\n\
         hint: use a floating-point dtype to preserve small values"
    )]
    Underflow { dtype: String, detail: String },

    /// A dtype string could not be recognized or is not supported.
    #[error("unknown or unsupported dtype: \"{0}\"")]
    UnknownDType(String),

    /// An operation does not support the given dtype.
    #[error(
        "operation '{op}' is not defined for dtype {dtype}\n\
         hint: cast to a compatible dtype first"
    )]
    UnsupportedDType { op: &'static str, dtype: String },

    /// Automatic type promotion between two dtypes is ambiguous.
    #[error(
        "type promotion between {lhs} and {rhs} is ambiguous — \
         specify the output dtype explicitly"
    )]
    AmbiguousPromotion { lhs: String, rhs: String },

    // -------------------------------------------------------------------------
    // Index & slice errors  (3xxx)
    // -------------------------------------------------------------------------

    /// An integer index is outside the valid range for its axis.
    #[error(
        "index {index} is out of bounds for axis {axis} with size {size}\n\
         hint: valid indices are -{size}..{size}"
    )]
    IndexOutOfBounds { index: i64, axis: usize, size: usize },

    /// More indices were provided than the array has dimensions.
    #[error(
        "too many indices: array is {ndim}D but {given} indices were given\n\
         hint: use None/newaxis to add dimensions rather than extra indices"
    )]
    TooManyIndices { given: usize, ndim: usize },

    /// A slice was constructed with a step of zero, which is undefined.
    #[error("slice step cannot be zero")]
    ZeroSliceStep,

    /// A slice range is out of bounds for the axis it indexes.
    #[error("slice [{start}:{stop}:{step}] is invalid for axis with size {size}")]
    SliceOutOfBounds { start: i64, stop: i64, step: i64, size: usize },

    /// A boolean mask has a different shape than the array it indexes.
    #[error(
        "boolean index shape {index_shape:?} does not match array shape {array_shape:?}\n\
         hint: the boolean mask must have the same shape as the array it indexes"
    )]
    BoolIndexShapeMismatch {
        index_shape: Vec<usize>,
        array_shape: Vec<usize>,
    },

    /// A fancy (integer-array) index contains a value outside the axis range.
    #[error(
        "fancy index on axis {axis} is out of bounds: \
         index value {index} exceeds axis size {size}"
    )]
    FancyIndexOutOfBounds { index: i64, axis: usize, size: usize },

    // -------------------------------------------------------------------------
    // Buffer & memory errors  (4xxx)
    // -------------------------------------------------------------------------

    /// A memory allocation request failed (likely OOM).
    #[error(
        "memory allocation failed: requested {bytes} bytes ({human})\n\
         hint: the system may be out of memory, or the requested size is unreasonable"
    )]
    AllocationFailed { bytes: usize, human: String },

    /// A pointer does not satisfy the alignment requirement for the element type.
    #[error(
        "pointer alignment error: operation requires {required}-byte alignment, \
         but the pointer is only {got}-byte aligned"
    )]
    AlignmentError { required: usize, got: usize },

    /// The provided buffer is smaller than what the operation needs.
    #[error(
        "buffer too small: operation requires {required} bytes, \
         but the buffer only holds {got} bytes"
    )]
    BufferTooSmall { required: usize, got: usize },

    /// A stride value is invalid for the given element size.
    #[error(
        "invalid stride on axis {axis}: stride {stride} is not a multiple \
         of the element size {element_size}"
    )]
    InvalidStride { axis: usize, stride: isize, element_size: usize },

    /// The combination of shape and strides would produce overlapping elements.
    #[error(
        "strides {strides:?} would cause overlapping elements for shape {shape:?} \
         with element size {element_size} — this would allow aliased mutable access"
    )]
    OverlappingStrides {
        shape: Vec<usize>,
        strides: Vec<isize>,
        element_size: usize,
    },

    /// An operation requires a contiguous (C-order) memory layout.
    #[error(
        "this operation requires a contiguous (C-order) array\n\
         hint: call .contiguous() to get a contiguous copy"
    )]
    NonContiguous,

    /// A mutation was attempted on a read-only array view.
    #[error(
        "this operation requires a writeable array, but the array is read-only\n\
         hint: call .to_owned() to get a mutable copy"
    )]
    ReadOnly,

    /// A resize or reallocation was attempted on an array that shares memory.
    #[error(
        "cannot resize or reallocate an array that shares memory with another array\n\
         hint: call .to_owned() to get an independent copy first"
    )]
    CannotResizeShared,

    /// An arithmetic overflow occurred while computing a buffer byte offset.
    #[error(
        "integer overflow computing buffer offset for shape {shape:?}, \
         strides {strides:?}, index {index:?}"
    )]
    OffsetOverflow {
        shape: Vec<usize>,
        strides: Vec<isize>,
        index: Vec<usize>,
    },

    // -------------------------------------------------------------------------
    // Compute / math errors  (5xxx)
    // -------------------------------------------------------------------------

    /// A matrix is singular (rank-deficient) and cannot be inverted or solved.
    #[error(
        "singular matrix: rank-deficient and cannot be inverted or solved\n\
         hint: use lstsq() for least-squares solutions of rank-deficient systems"
    )]
    SingularMatrix,

    /// An iterative solver did not converge within the allowed iterations.
    #[error(
        "iterative solver did not converge after {iterations} iterations \
         (tolerance = {tolerance:.3e}, final residual = {residual:.3e})\n\
         hint: increase max_iter or tolerance, or check the problem conditioning"
    )]
    NonConvergence {
        iterations: usize,
        tolerance: f64,
        residual: f64,
    },

    /// An operation is mathematically undefined for the given input.
    #[error("'{op}' is mathematically undefined: {reason}")]
    DomainError { op: &'static str, reason: String },

    /// A division by zero was detected.
    #[error(
        "division by zero\n\
         hint: use nan_to_num() to replace NaN/Inf results, or check divisors beforehand"
    )]
    DivisionByZero,

    /// Matrix dimensions are incompatible for a linear-algebra operation.
    #[error(
        "matrix dimensions incompatible for '{op}': \
         lhs is {lhs_rows}×{lhs_cols}, rhs is {rhs_rows}×{rhs_cols}"
    )]
    MatrixDimensionMismatch {
        op: &'static str,
        lhs_rows: usize,
        lhs_cols: usize,
        rhs_rows: usize,
        rhs_cols: usize,
    },

    /// Eigenvalue decomposition failed due to matrix properties.
    #[error(
        "eigenvalue decomposition failed: matrix is not {kind}\n\
         hint: check for NaN/Inf values or extreme conditioning"
    )]
    EigenDecompositionFailed { kind: &'static str },

    /// Cholesky decomposition failed because the matrix is not positive definite.
    #[error(
        "Cholesky decomposition failed: matrix is not positive definite\n\
         hint: add a small diagonal regularisation (ridge = ε·I) to make it SPD"
    )]
    NotPositiveDefinite,

    /// QR decomposition found a rank-deficient matrix when full rank was expected.
    #[error(
        "QR decomposition failed: matrix has rank {actual}, expected full rank {expected}"
    )]
    QRRankDeficient { expected: usize, actual: usize },

    /// SVD iteration did not converge.
    #[error("SVD did not converge after {iterations} iterations")]
    SVDNonConvergence { iterations: usize },

    /// The requested norm order is not supported for the given array dimensionality.
    #[error(
        "norm order '{order}' is not supported for {ndim}D arrays\n\
         hint: supported matrix norms are 1, 2, inf, fro, nuc"
    )]
    UnsupportedNormOrder { order: String, ndim: usize },

    // -------------------------------------------------------------------------
    // I/O errors  (6xxx)
    // -------------------------------------------------------------------------

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error(
        "invalid {format} magic bytes: expected {expected:?}, got {got:?}\n\
         hint: the file may be corrupt or not a valid {format} file"
    )]
    InvalidMagic {
        format: &'static str,
        expected: Vec<u8>,
        got: Vec<u8>,
    },

    #[error(
        "unsupported {format} version {major}.{minor} \
         (mohu supports up to {max_major}.{max_minor})"
    )]
    UnsupportedVersion {
        format: &'static str,
        major: u8,
        minor: u8,
        max_major: u8,
        max_minor: u8,
    },

    #[error("corrupt or malformed {format} data: {detail}")]
    CorruptData { format: &'static str, detail: String },

    #[error("unexpected end of file while reading {format} at byte offset {offset}")]
    UnexpectedEof { format: &'static str, offset: u64 },

    #[error(
        "unsupported compression codec '{codec}' in {format}\n\
         hint: supported codecs are: {supported}"
    )]
    UnsupportedCodec {
        format: &'static str,
        codec: String,
        supported: String,
    },

    #[error("NPY header parse error: {detail}")]
    NpyHeaderError { detail: String },

    #[error("NPZ archive has no entry named '{name}'")]
    NpzEntryNotFound { name: String },

    #[error("CSV parse error at row {row}, column {col}: {detail}")]
    CsvParseError { row: usize, col: usize, detail: String },

    // -------------------------------------------------------------------------
    // DLPack errors  (7xxx)
    // -------------------------------------------------------------------------

    #[error(
        "DLPack: unsupported device type {device_type}\n\
         hint: mohu supports CPU (device_type=1) only; \
         move the tensor to CPU first"
    )]
    DLPackUnsupportedDevice { device_type: i32 },

    #[error(
        "DLPack: version mismatch — mohu supports protocol {supported_major}.x, \
         tensor requests {got_major}.{got_minor}"
    )]
    DLPackVersionMismatch {
        supported_major: u32,
        got_major: u32,
        got_minor: u32,
    },

    #[error("DLPack: received a null DLManagedTensor pointer")]
    DLPackNullPointer,

    #[error(
        "DLPack: DLDataType (code={code}, bits={bits}, lanes={lanes}) \
         has no mohu equivalent"
    )]
    DLPackUnsupportedDType { code: u8, bits: u8, lanes: u16 },

    #[error("DLPack: {0}")]
    DLPackInvalid(String),

    // -------------------------------------------------------------------------
    // Arrow errors  (8xxx)
    // -------------------------------------------------------------------------

    #[error("Arrow schema mismatch: {0}")]
    ArrowSchema(String),

    #[error("Arrow IPC error: {0}")]
    ArrowIpc(String),

    #[error(
        "Arrow data type '{arrow_type}' has no mohu equivalent\n\
         hint: cast the Arrow column to a supported numeric type before conversion"
    )]
    ArrowUnsupportedType { arrow_type: String },

    #[error(
        "Arrow validity bitmap is inconsistent with array length {length}: {detail}"
    )]
    ArrowValidityError { length: usize, detail: String },

    // -------------------------------------------------------------------------
    // Python / PyO3 errors  (9xxx)
    // -------------------------------------------------------------------------

    #[error("Python type error: expected {expected}, got {got}")]
    PythonType { expected: &'static str, got: String },

    #[error("Python value error: {0}")]
    PythonValue(String),

    #[error("Python buffer protocol error: {0}")]
    PythonBuffer(String),

    #[error(
        "Python object does not implement the buffer protocol\n\
         hint: pass a NumPy array, bytes, bytearray, or a mohu array"
    )]
    PythonNoBuffer,

    #[error(
        "Python buffer has unsupported format string '{format}'\n\
         hint: mohu supports numeric buffer formats: b B h H i I l L q Q f d"
    )]
    PythonUnsupportedBufferFormat { format: String },

    // -------------------------------------------------------------------------
    // Contextual / structural errors  (10xxx)
    // -------------------------------------------------------------------------

    /// Wraps a lower-level `MohuError` with a human-readable context string.
    /// Use the [`ResultExt`](crate::context::ResultExt) trait instead of
    /// constructing this variant directly.
    #[error("{context}: {source}")]
    Context {
        context: String,
        #[source]
        source: Box<MohuError>,
    },

    #[error(
        "not yet implemented: {0}\n\
         contributions welcome at https://github.com/mohu-org/mohu"
    )]
    NotImplemented(&'static str),

    #[error(
        "internal error (this is a bug in mohu): {0}\n\
         please report at https://github.com/mohu-org/mohu/issues \
         with a minimal reproducer"
    )]
    Internal(String),

    /// Multiple errors accumulated in a single pass.
    ///
    /// Constructed by [`MultiError::into_result`](crate::multi::MultiError::into_result)
    /// when more than one error was collected.
    #[error("{0}")]
    Multiple(crate::multi::MultiError),
}

impl MohuError {
    /// Returns the numeric [`ErrorCode`](crate::codes::ErrorCode) for this variant.
    ///
    /// Error codes are stable across minor versions and allow programmatic
    /// branching without exhaustive `match` arms.
    pub fn code(&self) -> crate::codes::ErrorCode {
        use crate::codes::ErrorCode;
        match self {
            Self::ShapeMismatch { .. }             => ErrorCode::ShapeMismatch,
            Self::BroadcastError { .. }            => ErrorCode::BroadcastError,
            Self::DimensionMismatch { .. }         => ErrorCode::DimensionMismatch,
            Self::AxisOutOfRange { .. }            => ErrorCode::AxisOutOfRange,
            Self::ScalarArray                      => ErrorCode::ScalarArray,
            Self::ZeroSizedDimension { .. }        => ErrorCode::ZeroSizedDimension,
            Self::ShapeOverflow { .. }             => ErrorCode::ShapeOverflow,
            Self::ReshapeIncompatible { .. }       => ErrorCode::ReshapeIncompatible,
            Self::EmptyStackSequence               => ErrorCode::EmptyStackSequence,
            Self::ConcatShapeMismatch { .. }       => ErrorCode::ConcatShapeMismatch,

            Self::DTypeMismatch { .. }             => ErrorCode::DTypeMismatch,
            Self::InvalidCast { .. }               => ErrorCode::InvalidCast,
            Self::Overflow { .. }                  => ErrorCode::Overflow,
            Self::Underflow { .. }                 => ErrorCode::Underflow,
            Self::UnknownDType(_)                  => ErrorCode::UnknownDType,
            Self::UnsupportedDType { .. }          => ErrorCode::UnsupportedDType,
            Self::AmbiguousPromotion { .. }        => ErrorCode::AmbiguousPromotion,

            Self::IndexOutOfBounds { .. }          => ErrorCode::IndexOutOfBounds,
            Self::TooManyIndices { .. }            => ErrorCode::TooManyIndices,
            Self::ZeroSliceStep                    => ErrorCode::ZeroSliceStep,
            Self::SliceOutOfBounds { .. }          => ErrorCode::SliceOutOfBounds,
            Self::BoolIndexShapeMismatch { .. }    => ErrorCode::BoolIndexShapeMismatch,
            Self::FancyIndexOutOfBounds { .. }     => ErrorCode::FancyIndexOutOfBounds,

            Self::AllocationFailed { .. }          => ErrorCode::AllocationFailed,
            Self::AlignmentError { .. }            => ErrorCode::AlignmentError,
            Self::BufferTooSmall { .. }            => ErrorCode::BufferTooSmall,
            Self::InvalidStride { .. }             => ErrorCode::InvalidStride,
            Self::OverlappingStrides { .. }        => ErrorCode::OverlappingStrides,
            Self::NonContiguous                    => ErrorCode::NonContiguous,
            Self::ReadOnly                         => ErrorCode::ReadOnly,
            Self::CannotResizeShared               => ErrorCode::CannotResizeShared,
            Self::OffsetOverflow { .. }            => ErrorCode::OffsetOverflow,

            Self::SingularMatrix                   => ErrorCode::SingularMatrix,
            Self::NonConvergence { .. }            => ErrorCode::NonConvergence,
            Self::DomainError { .. }               => ErrorCode::DomainError,
            Self::DivisionByZero                   => ErrorCode::DivisionByZero,
            Self::MatrixDimensionMismatch { .. }   => ErrorCode::MatrixDimensionMismatch,
            Self::EigenDecompositionFailed { .. }  => ErrorCode::EigenDecompositionFailed,
            Self::NotPositiveDefinite              => ErrorCode::NotPositiveDefinite,
            Self::QRRankDeficient { .. }           => ErrorCode::QRRankDeficient,
            Self::SVDNonConvergence { .. }         => ErrorCode::SVDNonConvergence,
            Self::UnsupportedNormOrder { .. }      => ErrorCode::UnsupportedNormOrder,

            Self::Io(_)                            => ErrorCode::Io,
            Self::InvalidMagic { .. }              => ErrorCode::InvalidMagic,
            Self::UnsupportedVersion { .. }        => ErrorCode::UnsupportedVersion,
            Self::CorruptData { .. }               => ErrorCode::CorruptData,
            Self::UnexpectedEof { .. }             => ErrorCode::UnexpectedEof,
            Self::UnsupportedCodec { .. }          => ErrorCode::UnsupportedCodec,
            Self::NpyHeaderError { .. }            => ErrorCode::NpyHeaderError,
            Self::NpzEntryNotFound { .. }          => ErrorCode::NpzEntryNotFound,
            Self::CsvParseError { .. }             => ErrorCode::CsvParseError,

            Self::DLPackUnsupportedDevice { .. }   => ErrorCode::DLPackUnsupportedDevice,
            Self::DLPackVersionMismatch { .. }     => ErrorCode::DLPackVersionMismatch,
            Self::DLPackNullPointer                => ErrorCode::DLPackNullPointer,
            Self::DLPackUnsupportedDType { .. }    => ErrorCode::DLPackUnsupportedDType,
            Self::DLPackInvalid(_)                 => ErrorCode::DLPackInvalid,

            Self::ArrowSchema(_)                   => ErrorCode::ArrowSchema,
            Self::ArrowIpc(_)                      => ErrorCode::ArrowIpc,
            Self::ArrowUnsupportedType { .. }      => ErrorCode::ArrowUnsupportedType,
            Self::ArrowValidityError { .. }        => ErrorCode::ArrowValidityError,

            Self::PythonType { .. }                => ErrorCode::PythonType,
            Self::PythonValue(_)                   => ErrorCode::PythonValue,
            Self::PythonBuffer(_)                  => ErrorCode::PythonBuffer,
            Self::PythonNoBuffer                   => ErrorCode::PythonNoBuffer,
            Self::PythonUnsupportedBufferFormat{..}=> ErrorCode::PythonUnsupportedBufferFormat,

            Self::Context { .. }                   => ErrorCode::Context,
            Self::NotImplemented(_)                => ErrorCode::NotImplemented,
            Self::Internal(_)                      => ErrorCode::Internal,
            Self::Multiple(_)                      => ErrorCode::Internal,
        }
    }

    /// Returns the coarse [`ErrorKind`](crate::kind::ErrorKind) for this error.
    ///
    /// For `Context`-wrapped errors, delegates to the root cause.
    pub fn kind(&self) -> crate::kind::ErrorKind {
        use crate::kind::ErrorKind;
        match crate::chain::ErrorChain::root(self) {
            // Internal errors are always Fatal-kind.
            Self::Internal(_) => ErrorKind::Internal,
            // Multiple: take the worst kind across all inner errors.
            Self::Multiple(m) => m.iter()
                .map(|e| e.kind())
                .max_by_key(|k| *k as u8)
                .unwrap_or(ErrorKind::Internal),
            other => ErrorKind::from(other.code()),
        }
    }

    /// Returns `true` if this error is transient and the operation could
    /// reasonably be retried (e.g. allocation failure on OOM, I/O errors).
    pub fn is_transient(&self) -> bool {
        matches!(self, Self::Io(_) | Self::AllocationFailed { .. })
    }

    /// Returns `true` if the error is a caller programming mistake (wrong
    /// shape, wrong dtype, out-of-bounds index, etc.) rather than a runtime
    /// or environmental failure.
    pub fn is_usage_error(&self) -> bool {
        use crate::codes::ErrorCode as C;
        matches!(
            self.code(),
            C::ShapeMismatch
                | C::BroadcastError
                | C::DimensionMismatch
                | C::AxisOutOfRange
                | C::DTypeMismatch
                | C::InvalidCast
                | C::IndexOutOfBounds
                | C::TooManyIndices
                | C::ZeroSliceStep
                | C::SliceOutOfBounds
                | C::ReshapeIncompatible
                | C::BoolIndexShapeMismatch
                | C::FancyIndexOutOfBounds
        )
    }

    // -------------------------------------------------------------------------
    // Convenience constructors
    // -------------------------------------------------------------------------

    /// Builds an [`AllocationFailed`](Self::AllocationFailed) error with a
    /// human-readable size string computed automatically.
    pub fn alloc(bytes: usize) -> Self {
        Self::AllocationFailed { bytes, human: fmt_bytes(bytes) }
    }

    /// Builds an [`Internal`](Self::Internal) error. Use for assertion-style
    /// guards on invariants that should never be violated.
    pub fn bug(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Builds a [`DomainError`](Self::DomainError).
    pub fn domain(op: &'static str, reason: impl Into<String>) -> Self {
        Self::DomainError { op, reason: reason.into() }
    }

    /// Builds a [`MatrixDimensionMismatch`](Self::MatrixDimensionMismatch).
    pub fn matmul_shape(op: &'static str, lhs: [usize; 2], rhs: [usize; 2]) -> Self {
        Self::MatrixDimensionMismatch {
            op,
            lhs_rows: lhs[0],
            lhs_cols: lhs[1],
            rhs_rows: rhs[0],
            rhs_cols: rhs[1],
        }
    }
}

// ─── internal helpers ────────────────────────────────────────────────────────

fn fmt_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS[i])
}

// ─── Send + Sync assertions ───────────────────────────────────────────────────

// MohuError must be Send + Sync so it can be used across rayon thread pools
// and stored in Arc<Mutex<_>>. This compile-time assertion will fail if any
// variant accidentally introduces a non-Send or non-Sync field.
const _ASSERT_SEND_SYNC: () = {
    const fn _check<T: Send + Sync>() {}
    _check::<MohuError>();
};
