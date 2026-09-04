use mohu_buffer::{Buffer, MohuError};
use mohu_dtype::dtype::DType;

fn matrix<T: mohu_dtype::scalar::Scalar>(values: &[T], shape: &[usize]) -> Buffer {
    Buffer::from_slice(values).unwrap().reshape(shape).unwrap()
}

#[test]
fn concatenate_handles_axes_and_exact_order() {
    let a = matrix(&[1_i32, 2, 3, 4], &[2, 2]);
    let b = matrix(&[5_i32, 6], &[1, 2]);
    let out = Buffer::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 2, 3, 4, 5, 6]);

    let c = matrix(&[7_i32, 8, 9, 10], &[2, 2]);
    let wide = Buffer::concatenate(&[&a, &c], 1).unwrap();
    assert_eq!(wide.shape(), &[2, 4]);
    assert_eq!(wide.to_vec::<i32>().unwrap(), vec![1, 2, 7, 8, 3, 4, 9, 10]);
}

#[test]
fn concatenate_promotes_and_accepts_strided_views() {
    let base = matrix(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let transposed = base.transpose();
    let other = matrix(&[5.0_f64, 6.0], &[2, 1]);
    let out = Buffer::concatenate(&[&transposed, &other], 1).unwrap();
    assert_eq!(out.dtype(), DType::F64);
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.to_vec::<f64>().unwrap(),
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn concatenate_uses_current_typed_errors() {
    let a = matrix(&[1_i32, 2], &[1, 2]);
    assert!(matches!(
        Buffer::concatenate(&[], 0),
        Err(MohuError::EmptyStackSequence)
    ));
    assert!(matches!(
        Buffer::concatenate(&[&a], 2),
        Err(MohuError::AxisOutOfRange { .. })
    ));
    let b = matrix(&[3_i32, 4, 5], &[1, 3]);
    assert!(matches!(
        Buffer::concatenate(&[&a, &b], 0),
        Err(MohuError::ConcatShapeMismatch { .. })
    ));
    let scalar = Buffer::from_slice(&[1_i32]).unwrap().reshape(&[]).unwrap();
    assert!(matches!(
        Buffer::concatenate(&[&scalar], 0),
        Err(MohuError::ScalarArray)
    ));
}

#[test]
fn stack_inserts_axis_and_preserves_order() {
    let a = matrix(&[1_i32, 2, 3, 4], &[2, 2]);
    let b = matrix(&[5_i32, 6, 7, 8], &[2, 2]);
    for (axis, shape, expected) in [
        (0, vec![2, 2, 2], vec![1, 2, 3, 4, 5, 6, 7, 8]),
        (1, vec![2, 2, 2], vec![1, 2, 5, 6, 3, 4, 7, 8]),
        (2, vec![2, 2, 2], vec![1, 5, 2, 6, 3, 7, 4, 8]),
    ] {
        let out = Buffer::stack(&[&a, &b], axis).unwrap();
        assert_eq!(out.shape(), shape.as_slice());
        assert_eq!(out.to_vec::<i32>().unwrap(), expected);
    }
}

#[test]
fn stack_supports_scalar_and_strided_inputs() {
    let scalar_a = Buffer::from_slice(&[2_i32]).unwrap().reshape(&[]).unwrap();
    let scalar_b = Buffer::from_slice(&[3_i32]).unwrap().reshape(&[]).unwrap();
    let scalars = Buffer::stack(&[&scalar_a, &scalar_b], 0).unwrap();
    assert_eq!(scalars.shape(), &[2]);
    assert_eq!(scalars.to_vec::<i32>().unwrap(), vec![2, 3]);

    let base = matrix(&[1_i32, 2, 3, 4], &[2, 2]);
    let view = base.transpose();
    let out = Buffer::stack(&[&view, &view], 0).unwrap();
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 3, 2, 4, 1, 3, 2, 4]);
}

#[test]
fn empty_combine_results_are_valid() {
    let a = matrix::<i32>(&[], &[0, 3]);
    let b = matrix::<i32>(&[], &[0, 3]);
    let concat = Buffer::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(concat.shape(), &[0, 3]);
    assert_eq!(concat.len(), 0);

    let stack = Buffer::stack(&[&a, &b], 1).unwrap();
    assert_eq!(stack.shape(), &[0, 2, 3]);
    assert_eq!(stack.len(), 0);
}

#[test]
fn combined_output_is_new_owning_c_buffer() {
    let a = matrix(&[1_i32, 2], &[2]);
    let out = Buffer::concatenate(&[&a], 0).unwrap();
    assert!(out.is_c_contiguous());
    assert!(!out.is_shared());
    assert_eq!(out.to_vec::<i32>().unwrap(), vec![1, 2]);
}

#[test]
fn concatenate_preserves_complex_values_and_promotes_float_types() {
    use num_complex::Complex;
    let a = matrix(&[Complex::new(1.0_f32, 2.0), Complex::new(3.0, 4.0)], &[2]);
    let b = matrix(&[Complex::new(5.0_f32, 6.0)], &[1]);
    let out = Buffer::concatenate(&[&a, &b], 0).unwrap();
    assert_eq!(out.dtype(), DType::C64);
    assert_eq!(
        out.to_vec::<Complex<f32>>().unwrap(),
        vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ]
    );

    let low = Buffer::from_slice(&[1.0_f32, 2.0]).unwrap();
    let high = Buffer::from_slice(&[3.0_f64, 4.0]).unwrap();
    let stacked = Buffer::stack(&[&low, &high], 0).unwrap();
    assert_eq!(stacked.dtype(), DType::F64);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
