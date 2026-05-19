/// PyO3 error conversions — maps every `MohuError` variant to the most
/// semantically appropriate Python built-in exception.
///
/// This module is only compiled when the `python` feature is enabled.
/// The `mohu-py` crate enables this feature in its `Cargo.toml`.
///
/// # Mapping table
///
/// | MohuError variant(s)                       | Python exception  |
/// |--------------------------------------------|-------------------|
/// | Index*, Slice*, BoolIndex*, FancyIndex*     | `IndexError`      |
/// | Shape*, DType*, Compute*, Broadcast*        | `ValueError`      |
/// | PythonType                                  | `TypeError`       |
/// | AllocationFailed, Alignment*, BufferTooSmall| `MemoryError`     |
/// | Io*, InvalidMagic*, Corrupt*, UnexpectedEof | `OSError`         |
/// | NotImplemented                              | `NotImplementedError` |
/// | ReadOnly, NonContiguous                     | `BufferError`     |
/// | everything else                             | `RuntimeError`    |
use pyo3::exceptions::{
    PyBufferError, PyIndexError, PyMemoryError, PyNotImplementedError, PyOSError, PyRuntimeError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;

use crate::MohuError;

impl From<MohuError> for PyErr {
    fn from(err: MohuError) -> PyErr {
        let msg = err.to_string();
        match &err {
            // ── IndexError ────────────────────────────────────────────────
            MohuError::IndexOutOfBounds { .. }
            | MohuError::TooManyIndices { .. }
            | MohuError::ZeroSliceStep
            | MohuError::SliceOutOfBounds { .. }
            | MohuError::BoolIndexShapeMismatch { .. }
            | MohuError::FancyIndexOutOfBounds { .. } => PyIndexError::new_err(msg),

            // ── ValueError ────────────────────────────────────────────────
            MohuError::ShapeMismatch { .. }
            | MohuError::BroadcastError { .. }
            | MohuError::DimensionMismatch { .. }
            | MohuError::AxisOutOfRange { .. }
            | MohuError::ScalarArray
            | MohuError::ZeroSizedDimension { .. }
            | MohuError::ShapeOverflow { .. }
            | MohuError::ReshapeIncompatible { .. }
            | MohuError::EmptyStackSequence
            | MohuError::ConcatShapeMismatch { .. }
            | MohuError::DTypeMismatch { .. }
            | MohuError::InvalidCast { .. }
            | MohuError::Overflow { .. }
            | MohuError::Underflow { .. }
            | MohuError::UnknownDType(_)
            | MohuError::UnsupportedDType { .. }
            | MohuError::AmbiguousPromotion { .. }
            | MohuError::SingularMatrix
            | MohuError::NonConvergence { .. }
            | MohuError::DomainError { .. }
            | MohuError::DivisionByZero
            | MohuError::MatrixDimensionMismatch { .. }
            | MohuError::EigenDecompositionFailed { .. }
            | MohuError::NotPositiveDefinite
            | MohuError::QRRankDeficient { .. }
            | MohuError::SVDNonConvergence { .. }
            | MohuError::UnsupportedNormOrder { .. }
            | MohuError::PythonValue(_)
            | MohuError::DLPackUnsupportedDevice { .. }
            | MohuError::DLPackVersionMismatch { .. }
            | MohuError::DLPackUnsupportedDType { .. }
            | MohuError::DLPackInvalid(_)
            | MohuError::ArrowSchema(_)
            | MohuError::ArrowUnsupportedType { .. }
            | MohuError::ArrowValidityError { .. } => PyValueError::new_err(msg),

            // ── TypeError ─────────────────────────────────────────────────
            MohuError::PythonType { .. } | MohuError::PythonUnsupportedBufferFormat { .. } => {
                PyTypeError::new_err(msg)
            },

            // ── MemoryError ───────────────────────────────────────────────
            MohuError::AllocationFailed { .. }
            | MohuError::AlignmentError { .. }
            | MohuError::BufferTooSmall { .. }
            | MohuError::OffsetOverflow { .. }
            | MohuError::OverlappingStrides { .. } => PyMemoryError::new_err(msg),

            // ── BufferError ───────────────────────────────────────────────
            MohuError::NonContiguous
            | MohuError::ReadOnly
            | MohuError::CannotResizeShared
            | MohuError::InvalidStride { .. }
            | MohuError::PythonBuffer(_)
            | MohuError::PythonNoBuffer => PyBufferError::new_err(msg),

            // ── OSError ───────────────────────────────────────────────────
            MohuError::Io(_)
            | MohuError::InvalidMagic { .. }
            | MohuError::UnsupportedVersion { .. }
            | MohuError::CorruptData { .. }
            | MohuError::UnexpectedEof { .. }
            | MohuError::UnsupportedCodec { .. }
            | MohuError::NpyHeaderError { .. }
            | MohuError::NpzEntryNotFound { .. }
            | MohuError::CsvParseError { .. }
            | MohuError::ArrowIpc(_) => PyOSError::new_err(msg),

            // ── NotImplementedError ───────────────────────────────────────
            MohuError::NotImplemented(_) => PyNotImplementedError::new_err(msg),

            // ── RuntimeError (catch-all) ──────────────────────────────────
            MohuError::DLPackNullPointer
            | MohuError::Context { .. }
            | MohuError::Internal(_)
            | _ => PyRuntimeError::new_err(msg),
        }
    }
}
