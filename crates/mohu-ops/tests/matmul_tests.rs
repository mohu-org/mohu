use mohu_buffer::Buffer;
use mohu_dtype::dtype::DType;
use mohu_error::MohuError;
use mohu_ops::matmul::matmul;

fn matrix<T: mohu_dtype::scalar::Scalar>(values: &[T], shape: &[usize]) -> Buffer {
    Buffer::from_slice(values).unwrap().reshape(shape).unwrap()
}

#[test]
fn matmul_supports_all_rank_combinations() {
    let vector = Buffer::from_slice(&[1.0_f32, 2.0, 3.0]).unwrap();
    let matrix_rhs = matrix(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(
        matmul(&vector, &matrix_rhs)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![22.0, 28.0]
    );

    let matrix_lhs = matrix(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let rhs_vector = Buffer::from_slice(&[1.0_f32, 2.0, 3.0]).unwrap();
    assert_eq!(
        matmul(&matrix_lhs, &rhs_vector)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![14.0, 32.0]
    );

    let dot = matmul(&vector, &rhs_vector).unwrap();
    assert_eq!(dot.shape(), &[] as &[usize]);
    assert_eq!(dot.to_vec::<f32>().unwrap(), vec![14.0]);
}

#[test]
fn matmul_2d_and_rectangular() {
    let lhs = matrix(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let rhs = matrix(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
    assert_eq!(
        matmul(&lhs, &rhs).unwrap().to_vec::<f32>().unwrap(),
        vec![19.0, 22.0, 43.0, 50.0]
    );

    let wide = matrix(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let tall = matrix(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let out = matmul(&wide, &tall).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn matmul_validates_rank_and_shared_dimension() {
    let scalar = Buffer::from_slice(&[2.0_f32])
        .unwrap()
        .reshape(&[])
        .unwrap();
    let vector = Buffer::from_slice(&[1.0_f32, 2.0]).unwrap();
    assert!(matches!(
        matmul(&scalar, &vector),
        Err(MohuError::DimensionMismatch { .. })
    ));

    let lhs = matrix(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let rhs = matrix(&[1.0_f32, 2.0, 3.0], &[3, 1]);
    assert!(matches!(
        matmul(&lhs, &rhs),
        Err(MohuError::ShapeMismatch { .. })
    ));
}

#[test]
fn matmul_promotes_integer_inputs_to_f64() {
    let lhs = matrix(&[1_i32, 2, 3, 4], &[2, 2]);
    let rhs = matrix(&[5_i32, 6, 7, 8], &[2, 2]);
    let out = matmul(&lhs, &rhs).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    assert_eq!(out.to_vec::<f64>().unwrap(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_mixes_integer_and_float_inputs() {
    let lhs = matrix(&[1_i32, 2], &[1, 2]);
    let rhs = matrix(&[3.0_f32, 4.0], &[2, 1]);
    let out = matmul(&lhs, &rhs).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    assert_eq!(out.to_vec::<f64>().unwrap(), vec![11.0]);
}

#[test]
fn matmul_preserves_complex_arithmetic() {
    use num_complex::Complex;
    let lhs = matrix(
        &[Complex::new(1.0_f32, 2.0), Complex::new(3.0, 4.0)],
        &[1, 2],
    );
    let rhs = matrix(
        &[Complex::new(5.0_f32, 6.0), Complex::new(7.0, 8.0)],
        &[2, 1],
    );
    let out = matmul(&lhs, &rhs).unwrap();
    assert_eq!(out.dtype(), DType::C64);
    assert_eq!(
        out.to_vec::<Complex<f32>>().unwrap(),
        vec![Complex::new(-18.0, 68.0)]
    );
}

#[test]
fn matmul_accepts_non_contiguous_operands_and_zero_k() {
    let lhs_base = matrix(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let lhs = lhs_base.transpose();
    let rhs = matrix(&[5.0_f32, 6.0, 7.0, 8.0], &[2, 2]);
    assert_eq!(
        matmul(&lhs, &rhs).unwrap().to_vec::<f32>().unwrap(),
        vec![26.0, 30.0, 38.0, 44.0]
    );

    let empty_lhs = matrix::<f32>(&[], &[2, 0]);
    let empty_rhs = matrix::<f32>(&[], &[0, 3]);
    let out = matmul(&empty_lhs, &empty_rhs).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.to_vec::<f32>().unwrap(), vec![0.0; 6]);
}

#[test]
fn matmul_output_is_contiguous_and_owning() {
    let lhs = matrix(&[1.0_f64, 2.0], &[1, 2]);
    let rhs = matrix(&[3.0_f64, 4.0], &[2, 1]);
    let out = matmul(&lhs, &rhs).unwrap();
    assert!(out.is_c_contiguous());
    assert!(!out.is_shared());
}

#[test]
fn matmul_rejects_bool_inputs() {
    let lhs = Buffer::from_slice(&[true, false]).unwrap();
    let rhs = Buffer::from_slice(&[true, true]).unwrap();
    assert!(matches!(
        matmul(&lhs, &rhs),
        Err(MohuError::UnsupportedDType { .. })
    ));
}
