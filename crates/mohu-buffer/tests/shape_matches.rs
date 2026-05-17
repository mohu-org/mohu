use mohu_buffer::Buffer;
use mohu_dtype::DType;
use mohu_error::{MohuError, MohuResult};

#[test]
fn shape_matches_succeeds_for_equal_shapes() -> MohuResult<()> {
    let a = Buffer::zeros(DType::F64, &[2, 3])?;
    let b = Buffer::zeros(DType::F32, &[2, 3])?;
    a.shape_matches(&b)
}

#[test]
fn shape_matches_returns_shape_mismatch() {
    let a = Buffer::zeros(DType::F64, &[2, 3]).unwrap();
    let b = Buffer::zeros(DType::F64, &[3, 2]).unwrap();

    let err = a.shape_matches(&b).unwrap_err();
    match err {
        MohuError::ShapeMismatch { expected, got } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(got, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}
