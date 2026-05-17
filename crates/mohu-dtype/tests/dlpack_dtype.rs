use mohu_dtype::dlpack::DLDataTypeCode;
use mohu_dtype::{DType, ALL_DTYPES};
use mohu_error::MohuError;

#[test]
fn all_dtypes_round_trip_through_dlpack() {
    for dtype in ALL_DTYPES {
        if dtype == DType::Bool {
            // DLPack has no bool kind; Bool maps to kDLUInt(8) and round-trips as U8.
            continue;
        }
        let dl = dtype.to_dlpack();
        let (code, bits, lanes) = dl.to_raw();
        let back = DType::from_dlpack(code, bits, lanes).expect("round-trip");
        assert_eq!(back, dtype, "failed round-trip for {dtype}");
    }
}

#[test]
fn bool_maps_to_uint8_in_dlpack() {
    let dl = DType::Bool.to_dlpack();
    assert_eq!(dl.code, DLDataTypeCode::UInt);
    assert_eq!(dl.bits, 8);
    assert_eq!(dl.lanes, 1);
}

#[test]
fn f32_dlpack_encoding() {
    let dl = DType::F32.to_dlpack();
    assert_eq!(dl.code, DLDataTypeCode::Float);
    assert_eq!(dl.bits, 32);
    assert_eq!(dl.lanes, 1);
}

#[test]
fn c64_dlpack_encoding() {
    let dl = DType::C64.to_dlpack();
    assert_eq!(dl.code, DLDataTypeCode::Complex);
    assert_eq!(dl.bits, 64);
    assert_eq!(dl.lanes, 1);
}

#[test]
fn unknown_dlpack_code_is_rejected() {
    let err = DType::from_dlpack(99, 32, 1).unwrap_err();
    assert!(matches!(err, MohuError::DLPackInvalid(_)));
}

#[test]
fn unsupported_dlpack_dtype_is_rejected() {
    let err = DType::from_dlpack(2, 128, 1).unwrap_err();
    assert!(matches!(err, MohuError::DLPackUnsupportedDType { .. }));
}

#[test]
fn multi_lane_dlpack_is_rejected() {
    let err = DType::from_dlpack(2, 32, 4).unwrap_err();
    assert!(matches!(err, MohuError::DLPackInvalid(_)));
}
