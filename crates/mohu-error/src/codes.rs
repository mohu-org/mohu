/// Stable numeric error codes for every [`MohuError`](crate::MohuError) variant.
///
/// Error codes let downstream code branch on error category without doing
/// exhaustive pattern matching on the full enum. They are guaranteed to be
/// stable across minor versions — new variants get new codes, existing codes
/// never change meaning.
///
/// # Layout
///
/// | Range      | Domain          |
/// |------------|-----------------|
/// | 1000–1999  | Shape           |
/// | 2000–2999  | DType           |
/// | 3000–3999  | Index / slice   |
/// | 4000–4999  | Buffer / memory |
/// | 5000–5999  | Compute / math  |
/// | 6000–6999  | I/O             |
/// | 7000–7999  | DLPack          |
/// | 8000–8999  | Arrow           |
/// | 9000–9999  | Python / PyO3   |
/// | 10000+     | General         |
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, codes::ErrorCode};
/// let err = MohuError::SingularMatrix;
/// match err.code() {
///     ErrorCode::SingularMatrix => eprintln!("rank-deficient system"),
///     c if (c as u32) >= 5000 && (c as u32) < 6000 => eprintln!("some compute error"),
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[non_exhaustive]
pub enum ErrorCode {
    // Shape  (1xxx)
    ShapeMismatch = 1000,
    BroadcastError = 1001,
    DimensionMismatch = 1002,
    AxisOutOfRange = 1003,
    ScalarArray = 1004,
    ZeroSizedDimension = 1005,
    ShapeOverflow = 1006,
    ReshapeIncompatible = 1007,
    EmptyStackSequence = 1008,
    ConcatShapeMismatch = 1009,

    // DType  (2xxx)
    DTypeMismatch = 2000,
    InvalidCast = 2001,
    Overflow = 2002,
    Underflow = 2003,
    UnknownDType = 2004,
    UnsupportedDType = 2005,
    AmbiguousPromotion = 2006,

    // Index / slice  (3xxx)
    IndexOutOfBounds = 3000,
    TooManyIndices = 3001,
    ZeroSliceStep = 3002,
    SliceOutOfBounds = 3003,
    BoolIndexShapeMismatch = 3004,
    FancyIndexOutOfBounds = 3005,

    // Buffer / memory  (4xxx)
    AllocationFailed = 4000,
    AlignmentError = 4001,
    BufferTooSmall = 4002,
    InvalidStride = 4003,
    OverlappingStrides = 4004,
    NonContiguous = 4005,
    ReadOnly = 4006,
    CannotResizeShared = 4007,
    OffsetOverflow = 4008,

    // Compute / math  (5xxx)
    SingularMatrix = 5000,
    NonConvergence = 5001,
    DomainError = 5002,
    DivisionByZero = 5003,
    MatrixDimensionMismatch = 5004,
    EigenDecompositionFailed = 5005,
    NotPositiveDefinite = 5006,
    QRRankDeficient = 5007,
    SVDNonConvergence = 5008,
    UnsupportedNormOrder = 5009,

    // I/O  (6xxx)
    Io = 6000,
    InvalidMagic = 6001,
    UnsupportedVersion = 6002,
    CorruptData = 6003,
    UnexpectedEof = 6004,
    UnsupportedCodec = 6005,
    NpyHeaderError = 6006,
    NpzEntryNotFound = 6007,
    CsvParseError = 6008,

    // DLPack  (7xxx)
    DLPackUnsupportedDevice = 7000,
    DLPackVersionMismatch = 7001,
    DLPackNullPointer = 7002,
    DLPackUnsupportedDType = 7003,
    DLPackInvalid = 7004,

    // Arrow  (8xxx)
    ArrowSchema = 8000,
    ArrowIpc = 8001,
    ArrowUnsupportedType = 8002,
    ArrowValidityError = 8003,

    // Python / PyO3  (9xxx)
    PythonType = 9000,
    PythonValue = 9001,
    PythonBuffer = 9002,
    PythonNoBuffer = 9003,
    PythonUnsupportedBufferFormat = 9004,

    // General  (10xxx)
    Context = 10000,
    NotImplemented = 10001,
    Internal = 10002,
}

impl ErrorCode {
    /// Returns the domain name for this error code.
    pub fn domain(self) -> &'static str {
        match self as u32 {
            1000..=1999 => "shape",
            2000..=2999 => "dtype",
            3000..=3999 => "index",
            4000..=4999 => "buffer",
            5000..=5999 => "compute",
            6000..=6999 => "io",
            7000..=7999 => "dlpack",
            8000..=8999 => "arrow",
            9000..=9999 => "python",
            _ => "general",
        }
    }

    /// Returns `true` if this code falls in the shape domain (1000–1999).
    pub fn is_shape(self) -> bool {
        matches!(self as u32, 1000..=1999)
    }

    /// Returns `true` if this code falls in the dtype domain (2000–2999).
    pub fn is_dtype(self) -> bool {
        matches!(self as u32, 2000..=2999)
    }

    /// Returns `true` if this code falls in the index domain (3000–3999).
    pub fn is_index(self) -> bool {
        matches!(self as u32, 3000..=3999)
    }

    /// Returns `true` if this code falls in the buffer domain (4000–4999).
    pub fn is_buffer(self) -> bool {
        matches!(self as u32, 4000..=4999)
    }

    /// Returns `true` if this code falls in the compute domain (5000–5999).
    pub fn is_compute(self) -> bool {
        matches!(self as u32, 5000..=5999)
    }

    /// Returns `true` if this code falls in the I/O domain (6000–6999).
    pub fn is_io(self) -> bool {
        matches!(self as u32, 6000..=6999)
    }

    /// Returns `true` if this code falls in the DLPack domain (7000–7999).
    pub fn is_dlpack(self) -> bool {
        matches!(self as u32, 7000..=7999)
    }

    /// Returns `true` if this code falls in the Arrow domain (8000–8999).
    pub fn is_arrow(self) -> bool {
        matches!(self as u32, 8000..=8999)
    }

    /// Returns `true` if this code falls in the Python domain (9000–9999).
    pub fn is_python(self) -> bool {
        matches!(self as u32, 9000..=9999)
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "E{:04}", *self as u32)
    }
}
