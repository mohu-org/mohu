use mohu_error::{MohuError, MultiError, codes::ErrorCode};

fn error(message: &'static str) -> MohuError {
    MohuError::domain("test", message)
}

#[test]
fn empty_collector_is_ok() {
    assert!(MultiError::new().into_result().is_ok());
}

#[test]
fn one_error_is_returned_directly() {
    let mut errors = MultiError::new();
    errors.push(error("first"));
    let result = errors.into_result();
    assert!(
        matches!(result, Err(MohuError::DomainError { op: "test", ref reason }) if reason == "first")
    );
}

#[test]
fn multiple_errors_are_wrapped() {
    let mut errors = MultiError::new();
    errors.push(error("first"));
    errors.push(error("second"));
    assert!(matches!(errors.into_result(), Err(MohuError::Multiple(_))));
}

#[test]
fn iteration_preserves_insertion_order() {
    let mut errors = MultiError::new();
    errors.push(error("A"));
    errors.push(error("B"));
    errors.push(error("C"));
    let messages: Vec<_> = errors.iter().map(ToString::to_string).collect();
    assert!(messages[0].contains("A"));
    assert!(messages[1].contains("B"));
    assert!(messages[2].contains("C"));
}

#[test]
fn display_contains_count_and_messages() {
    let mut errors = MultiError::new();
    errors.push(error("left"));
    errors.push(error("right"));
    let text = errors.to_string();
    assert!(text.contains("2 error(s)"));
    assert!(text.contains("left"));
    assert!(text.contains("right"));
}

#[test]
fn collect_keeps_only_errors() {
    let mut errors = MultiError::new();
    errors.collect::<()>(Ok(()));
    errors.collect::<()>(Err(error("bad")));
    assert_eq!(errors.len(), 1);
    assert!(errors.has_errors());
}

#[test]
fn borrowed_into_iterator_matches_iter() {
    let mut errors = MultiError::new();
    errors.push(error("A"));
    errors.push(error("B"));
    assert_eq!((&errors).into_iter().count(), 2);
}

#[test]
fn error_codes_round_trip_and_reject_unknown() {
    let codes = [
        ErrorCode::ShapeMismatch,
        ErrorCode::BroadcastError,
        ErrorCode::DimensionMismatch,
        ErrorCode::AxisOutOfRange,
        ErrorCode::ScalarArray,
        ErrorCode::ZeroSizedDimension,
        ErrorCode::ShapeOverflow,
        ErrorCode::ReshapeIncompatible,
        ErrorCode::EmptyStackSequence,
        ErrorCode::ConcatShapeMismatch,
        ErrorCode::DTypeMismatch,
        ErrorCode::InvalidCast,
        ErrorCode::Overflow,
        ErrorCode::Underflow,
        ErrorCode::UnknownDType,
        ErrorCode::UnsupportedDType,
        ErrorCode::AmbiguousPromotion,
        ErrorCode::IndexOutOfBounds,
        ErrorCode::TooManyIndices,
        ErrorCode::ZeroSliceStep,
        ErrorCode::SliceOutOfBounds,
        ErrorCode::BoolIndexShapeMismatch,
        ErrorCode::FancyIndexOutOfBounds,
        ErrorCode::AllocationFailed,
        ErrorCode::AlignmentError,
        ErrorCode::BufferTooSmall,
        ErrorCode::InvalidStride,
        ErrorCode::OverlappingStrides,
        ErrorCode::NonContiguous,
        ErrorCode::ReadOnly,
        ErrorCode::CannotResizeShared,
        ErrorCode::OffsetOverflow,
        ErrorCode::SingularMatrix,
        ErrorCode::NonConvergence,
        ErrorCode::DomainError,
        ErrorCode::DivisionByZero,
        ErrorCode::MatrixDimensionMismatch,
        ErrorCode::EigenDecompositionFailed,
        ErrorCode::NotPositiveDefinite,
        ErrorCode::QRRankDeficient,
        ErrorCode::SVDNonConvergence,
        ErrorCode::UnsupportedNormOrder,
        ErrorCode::Io,
        ErrorCode::InvalidMagic,
        ErrorCode::UnsupportedVersion,
        ErrorCode::CorruptData,
        ErrorCode::UnexpectedEof,
        ErrorCode::UnsupportedCodec,
        ErrorCode::NpyHeaderError,
        ErrorCode::NpzEntryNotFound,
        ErrorCode::CsvParseError,
        ErrorCode::DLPackUnsupportedDevice,
        ErrorCode::DLPackVersionMismatch,
        ErrorCode::DLPackNullPointer,
        ErrorCode::DLPackUnsupportedDType,
        ErrorCode::DLPackInvalid,
        ErrorCode::ArrowSchema,
        ErrorCode::ArrowIpc,
        ErrorCode::ArrowUnsupportedType,
        ErrorCode::ArrowValidityError,
        ErrorCode::PythonType,
        ErrorCode::PythonValue,
        ErrorCode::PythonBuffer,
        ErrorCode::PythonNoBuffer,
        ErrorCode::PythonUnsupportedBufferFormat,
        ErrorCode::Context,
        ErrorCode::NotImplemented,
        ErrorCode::Internal,
    ];
    for code in codes {
        assert_eq!(ErrorCode::try_from(code as u32), Ok(code));
    }
    for raw in [999, 1010, 1999, 2007, 3999, 6009, 9999, 10003, u32::MAX] {
        assert_eq!(ErrorCode::try_from(raw), Err(()));
    }
    assert_eq!(ErrorCode::ShapeMismatch.to_string(), "E1000");
}
