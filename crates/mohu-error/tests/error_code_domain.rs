use mohu_error::ErrorCode;

#[test]
fn domain_returns_expected_names() {
    assert_eq!(ErrorCode::ShapeMismatch.domain(), "shape");
    assert_eq!(ErrorCode::DTypeMismatch.domain(), "dtype");
    assert_eq!(ErrorCode::Io.domain(), "io");
    assert_eq!(ErrorCode::Internal.domain(), "general");
}

#[test]
fn domain_helpers_match_ranges() {
    assert!(ErrorCode::ShapeMismatch.is_shape_error());
    assert!(ErrorCode::DTypeMismatch.is_dtype_error());
    assert!(ErrorCode::Io.is_io_error());
    assert!(!ErrorCode::ShapeMismatch.is_io_error());
}
