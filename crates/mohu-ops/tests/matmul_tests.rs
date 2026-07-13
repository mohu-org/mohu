use mohu_buffer::buffer::Buffer;
use mohu_ops::matmul::matmul;

#[test]
fn matmul_2x2() {
    let a = Buffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let b = Buffer::from_vec(vec![5.0f32, 6.0, 7.0, 8.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);

    let out = c.as_slice::<f32>().unwrap();

    mohu_testing::approx::assert_allclose(out, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn matmul_shape_mismatch() {
    let a = Buffer::from_vec(vec![1.0f32; 6])
        .unwrap()
        .reshape(&[2, 3])
        .unwrap();

    let b = Buffer::from_vec(vec![1.0f32; 8])
        .unwrap()
        .reshape(&[4, 2])
        .unwrap();

    assert!(matmul(&a, &b).is_err());
}

#[test]
fn matmul_dot_product() {
    let a = Buffer::from_vec(vec![1.0f32, 2.0, 3.0]).unwrap();
    let b = Buffer::from_vec(vec![4.0f32, 5.0, 6.0]).unwrap();

    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[] as &[usize]);

    let out = c.as_slice::<f32>().unwrap();

    assert_eq!(out, &[32.0]);
}

#[test]
fn matmul_row_vector_matrix() {
    let a = Buffer::from_vec(vec![1.0f32, 2.0]).unwrap();

    let b = Buffer::from_vec(vec![3.0f32, 4.0, 5.0, 6.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2]);

    let out = c.as_slice::<f32>().unwrap();

    assert_eq!(out, &[13.0, 16.0]);
}

#[test]
fn matmul_matrix_vector() {
    let a = Buffer::from_vec(vec![1.0f32, 2.0, 3.0, 4.0])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let b = Buffer::from_vec(vec![5.0f32, 6.0]).unwrap();

    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2]);

    let out = c.as_slice::<f32>().unwrap();

    assert_eq!(out, &[17.0, 39.0]);
}

use mohu_dtype::dtype::DType;

#[test]
fn matmul_integer_promotes_to_f64() {
    let a = Buffer::from_vec(vec![1i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let b = Buffer::from_vec(vec![5i32, 6, 7, 8])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();

    let c = matmul(&a, &b).unwrap();

    assert_eq!(c.dtype(), DType::F64);
}
