//! Property-based tests for core `Buffer` invariants (issue #31).

use mohu_buffer::{Buffer, SliceArg};
use mohu_dtype::DType;
use proptest::prelude::*;

fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        (1usize..=12).prop_map(|n| vec![n]),
        (1usize..=6, 1usize..=6).prop_map(|(r, c)| vec![r, c]),
        (1usize..=4, 1usize..=4, 1usize..=4).prop_map(|(a, b, c)| vec![a, b, c]),
    ]
}

fn arb_f64_buffer() -> impl Strategy<Value = Buffer> {
    arb_shape().prop_flat_map(|shape| {
        let n: usize = shape.iter().product();
        proptest::collection::vec(-50.0f64..50.0, n).prop_map(move |data| {
            Buffer::from_slice(&data)
                .unwrap()
                .reshape(&shape)
                .unwrap()
        })
    })
}

fn assert_bufs_eq(a: &Buffer, b: &Buffer) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.shape(), b.shape());
    prop_assert_eq!(a.dtype(), b.dtype());
    let a = a.to_contiguous().unwrap();
    let b = b.to_contiguous().unwrap();
    prop_assert_eq!(a.to_vec::<f64>().unwrap(), b.to_vec::<f64>().unwrap());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn from_slice_round_trips(data in proptest::collection::vec(-100i32..100, 1..=48)) {
        let buf = Buffer::from_slice(&data).unwrap();
        prop_assert_eq!(buf.to_vec::<i32>().unwrap(), data);
    }

    #[test]
    fn zeros_sums_to_zero(shape in arb_shape()) {
        let z = Buffer::zeros(DType::F64, &shape).unwrap();
        prop_assert_eq!(z.sum_all_f64().unwrap(), 0.0);
    }

    #[test]
    fn transpose_is_involution(buf in arb_f64_buffer().prop_filter("2d", |b| b.ndim() == 2)) {
        let tt = buf.transpose().transpose();
        assert_bufs_eq(&buf, &tt)?;
    }

    #[test]
    fn reshape_flatten_and_back(buf in arb_f64_buffer()) {
        let orig: Vec<usize> = buf.shape().to_vec();
        let flat = buf.reshape(&[buf.len()]).unwrap();
        let back = flat.reshape(&orig).unwrap();
        assert_bufs_eq(&buf, &back)?;
    }

    #[test]
    fn slice_axis_full_preserves_data(buf in arb_f64_buffer()) {
        for axis in 0..buf.ndim() {
            let sliced = buf.slice_axis(axis, SliceArg::FULL).unwrap();
            assert_bufs_eq(&buf, &sliced)?;
        }
    }

    #[test]
    fn flip_twice_is_identity(buf in arb_f64_buffer()) {
        for axis in 0..buf.ndim() {
            let restored = buf.flip(axis).unwrap().flip(axis).unwrap();
            assert_bufs_eq(&buf, &restored)?;
        }
    }

    #[test]
    fn sum_axis_reduces_to_total(buf in arb_f64_buffer()) {
        let total = buf.sum_all_f64().unwrap();
        let n = buf.len().max(1) as f64;
        for axis in 0..buf.ndim() {
            let partial = buf.sum_axis(axis, false).unwrap();
            let partial_sum = partial.sum_all_f64().unwrap();
            prop_assert!((partial_sum - total).abs() < 1e-9 * n);
        }
    }
}
