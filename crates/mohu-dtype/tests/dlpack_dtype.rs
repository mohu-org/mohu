use mohu_dtype::dlpack::DLDataTypeCode;
use mohu_dtype::{ALL_DTYPES, DType, MohuError};

#[test]
fn all_dtypes_round_trip_through_dlpack() {
    for dtype in ALL_DTYPES {
        let raw = dtype.to_dlpack().to_raw();
        assert_eq!(DType::from_dlpack(raw.0, raw.1, raw.2).unwrap(), dtype);
    }
}

#[test]
fn normative_raw_codes_are_stable() {
    assert_eq!(DType::Bool.to_dlpack().to_raw(), (6, 8, 1));
    assert_eq!(DType::U8.to_dlpack().to_raw(), (1, 8, 1));
    assert_eq!(DType::F32.to_dlpack().to_raw(), (2, 32, 1));
    assert_eq!(DType::BF16.to_dlpack().to_raw(), (4, 16, 1));
    assert_eq!(DType::C64.to_dlpack().to_raw(), (5, 64, 1));
    assert_eq!(DLDataTypeCode::from_u8(6).unwrap(), DLDataTypeCode::Bool);
    assert_eq!(DType::from_dlpack(6, 8, 1).unwrap(), DType::Bool);
    assert_eq!(DType::from_dlpack(1, 8, 1).unwrap(), DType::U8);
}

#[test]
fn malformed_dlpack_types_preserve_error_categories() {
    assert!(matches!(
        DType::from_dlpack(3, 32, 1),
        Err(MohuError::DLPackInvalid(_))
    ));
    assert!(matches!(
        DType::from_dlpack(6, 16, 1),
        Err(MohuError::DLPackUnsupportedDType { .. })
    ));
    assert!(matches!(
        DType::from_dlpack(2, 32, 0),
        Err(MohuError::DLPackInvalid(_))
    ));
    assert!(matches!(
        DType::from_dlpack(2, 32, 2),
        Err(MohuError::DLPackInvalid(_))
    ));
    assert!(matches!(
        DType::from_dlpack(5, 32, 1),
        Err(MohuError::DLPackUnsupportedDType { .. })
    ));
}
