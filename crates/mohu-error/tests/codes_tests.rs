use mohu_error::ErrorCode;

#[test]
fn try_from_known_codes() {
    let code = ErrorCode::try_from(1000u32).unwrap();
    assert_eq!(code, ErrorCode::ShapeMismatch);
    assert_eq!(ErrorCode::try_from(10002u32).unwrap(), ErrorCode::Internal);
}

#[test]
fn try_from_rejects_unknown_codes() {
    assert!(ErrorCode::try_from(999u32).is_err());
    assert!(ErrorCode::try_from(1500u32).is_err());
}

#[test]
fn display_includes_variant_and_number() {
    let code = ErrorCode::ShapeMismatch;
    assert_eq!(format!("{code}"), "ShapeMismatch (1000)");
}

#[test]
fn from_u32_roundtrip() {
    let code = ErrorCode::ShapeMismatch;
    assert_eq!(u32::from(code), 1000);
}
