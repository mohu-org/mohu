use mohu_buffer::{Buffer, ops};
use mohu_dtype::DType;
use mohu_error::MohuError;

fn assert_oob(error: MohuError, index: i64, size: usize) {
    assert!(matches!(
        error,
        MohuError::IndexOutOfBounds {
            index: actual,
            axis: 0,
            size: actual_size,
        } if actual == index && actual_size == size
    ));
}

#[test]
fn gather_supports_negative_and_repeated_indices() {
    let src = Buffer::from_slice(&[10_i32, 20, 30, 40, 50]).unwrap();
    let indices = Buffer::from_slice(&[0_i64, -1, 0]).unwrap();
    let mut dst = Buffer::zeros(DType::I32, &[3]).unwrap();

    ops::gather(&src, &indices, &mut dst).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[10, 50, 10]);
}

#[test]
fn gather_rejects_all_out_of_bounds_without_mutation() {
    for &index in &[5_i64, 6, -6] {
        let src = Buffer::from_slice(&[10_i32, 20, 30, 40, 50]).unwrap();
        let indices = Buffer::from_slice(&[0_i64, index, 1]).unwrap();
        let mut dst = Buffer::from_slice(&[99_i32, 99, 99]).unwrap();

        let error = ops::gather(&src, &indices, &mut dst).unwrap_err();
        assert_oob(error, index, 5);
        assert_eq!(dst.as_slice::<i32>().unwrap(), &[99, 99, 99]);
    }
}

#[test]
fn gather_empty_is_a_noop() {
    let src = Buffer::zeros(DType::I32, &[0]).unwrap();
    let indices: Buffer = Buffer::from_slice(&[] as &[i64]).unwrap();
    let mut dst = Buffer::zeros(DType::I32, &[0]).unwrap();

    ops::gather(&src, &indices, &mut dst).unwrap();
    assert_eq!(dst.len(), 0);
}

#[test]
fn gather_zero_length_rejects_an_actual_index() {
    let src = Buffer::zeros(DType::I32, &[0]).unwrap();
    let indices = Buffer::from_slice(&[0_i64]).unwrap();
    let mut dst = Buffer::zeros(DType::I32, &[1]).unwrap();

    let error = ops::gather(&src, &indices, &mut dst).unwrap_err();
    assert_oob(error, 0, 0);
}

#[test]
fn scatter_supports_negative_indices_and_last_write_wins() {
    let mut dst = Buffer::zeros(DType::I32, &[5]).unwrap();
    let indices = Buffer::from_slice(&[1_i64, 1, -1]).unwrap();
    let src = Buffer::from_slice(&[10_i32, 20, 30]).unwrap();

    ops::scatter(&mut dst, &indices, &src).unwrap();

    assert_eq!(dst.as_slice::<i32>().unwrap(), &[0, 20, 0, 0, 30]);
}

#[test]
fn scatter_rejects_all_out_of_bounds_without_mutation() {
    for &index in &[5_i64, 6, -6] {
        let mut dst = Buffer::from_slice(&[9_i32, 9, 9, 9, 9]).unwrap();
        let indices = Buffer::from_slice(&[0_i64, index, 1]).unwrap();
        let src = Buffer::from_slice(&[10_i32, 20, 30]).unwrap();

        let error = ops::scatter(&mut dst, &indices, &src).unwrap_err();
        assert_oob(error, index, 5);
        assert_eq!(dst.as_slice::<i32>().unwrap(), &[9, 9, 9, 9, 9]);
    }
}

#[test]
fn scatter_empty_is_a_noop() {
    let mut dst = Buffer::zeros(DType::I32, &[0]).unwrap();
    let indices: Buffer = Buffer::from_slice(&[] as &[i64]).unwrap();
    let src: Buffer = Buffer::from_slice(&[] as &[i32]).unwrap();

    ops::scatter(&mut dst, &indices, &src).unwrap();
    assert_eq!(dst.len(), 0);
}

#[test]
fn scatter_zero_length_rejects_an_actual_index() {
    let mut dst = Buffer::zeros(DType::I32, &[0]).unwrap();
    let indices = Buffer::from_slice(&[0_i64]).unwrap();
    let src = Buffer::from_slice(&[1_i32]).unwrap();

    let error = ops::scatter(&mut dst, &indices, &src).unwrap_err();
    assert_oob(error, 0, 0);
}

#[test]
fn scatter_then_gather_round_trips_unique_indices() {
    let mut dst = Buffer::zeros(DType::I32, &[5]).unwrap();
    let indices = Buffer::from_slice(&[0_i64, 2, 4]).unwrap();
    let values = Buffer::from_slice(&[7_i32, 8, 9]).unwrap();

    ops::scatter(&mut dst, &indices, &values).unwrap();
    let mut gathered = Buffer::zeros(DType::I32, &[3]).unwrap();
    ops::gather(&dst, &indices, &mut gathered).unwrap();

    assert_eq!(gathered.as_slice::<i32>().unwrap(), &[7, 8, 9]);
}
