//! Bounded, contract-driven property tests for implemented Buffer invariants.

use mohu_buffer::{Buffer, SliceArg};
use mohu_dtype::DType;
use proptest::prelude::*;

fn arb_shape() -> impl Strategy<Value = Vec<usize>> {
    prop_oneof![
        (1usize..=4).prop_map(|n| vec![n]),
        (1usize..=4, 1usize..=4).prop_map(|(a, b)| vec![a, b]),
        (1usize..=4, 1usize..=4, 1usize..=4).prop_map(|(a, b, c)| vec![a, b, c]),
    ]
}

fn arb_f64_buffer() -> impl Strategy<Value = Buffer> {
    arb_shape().prop_flat_map(|shape| {
        let len: usize = shape.iter().product();
        proptest::collection::vec(-50.0f64..50.0, len)
            .prop_map(move |data| Buffer::from_slice(&data).unwrap().reshape(&shape).unwrap())
    })
}

fn assert_logically_equal(a: &Buffer, b: &Buffer) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.shape(), b.shape());
    prop_assert_eq!(a.dtype(), b.dtype());
    let a = a.to_contiguous().unwrap().to_vec::<f64>().unwrap();
    let b = b.to_contiguous().unwrap().to_vec::<f64>().unwrap();
    prop_assert_eq!(a, b);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    // Non-empty rank 1..=3, <=64 elements; transpose reverses axes.
    #[test]
    fn transpose_twice_preserves_logical_buffer(buf in arb_f64_buffer()) {
        let restored = buf.transpose().transpose();
        assert_logically_equal(&buf, &restored)?;
    }

    // Valid same-element-count reshape on generated C-contiguous buffers.
    #[test]
    fn reshape_flatten_round_trip_preserves_values(buf in arb_f64_buffer()) {
        let original = buf.shape().to_vec();
        let flattened = buf.reshape(&[buf.len()]).unwrap();
        let restored = flattened.reshape(&original).unwrap();
        assert_logically_equal(&buf, &restored)?;
    }

    // FULL slice contract checks logical identity, not backing-layout identity.
    #[test]
    fn full_axis_slice_preserves_logical_buffer(buf in arb_f64_buffer()) {
        for axis in 0..buf.ndim() {
            let sliced = buf.slice_axis(axis, SliceArg::FULL).unwrap();
            assert_logically_equal(&buf, &sliced)?;
        }
    }

    #[test]
    fn flip_twice_preserves_logical_buffer(buf in arb_f64_buffer()) {
        for axis in 0..buf.ndim() {
            let restored = buf.flip(axis).unwrap().flip(axis).unwrap();
            assert_logically_equal(&buf, &restored)?;
        }
    }

    // Bounded F64 values avoid overflow; tolerance covers summation order.
    #[test]
    fn f64_axis_reductions_preserve_total_sum(buf in arb_f64_buffer()) {
        let expected = buf.sum_all_f64().unwrap();
        for axis in 0..buf.ndim() {
            let reduced = buf.sum_axis(axis, false).unwrap();
            let actual = reduced.sum_all_f64().unwrap();
            prop_assert!((actual - expected).abs() <= 1e-10 * (buf.len().max(1) as f64));
        }
    }

    #[test]
    fn from_slice_round_trip_preserves_values(data in proptest::collection::vec(-100i32..100, 0..=64)) {
        let buffer = Buffer::from_slice(&data).unwrap();
        prop_assert_eq!(buffer.to_vec::<i32>().unwrap(), data);
    }

    #[test]
    fn zeros_have_zero_storage(shape in arb_shape()) {
        let buffer = Buffer::zeros(DType::F64, &shape).unwrap();
        let values = buffer.to_contiguous().unwrap().to_vec::<f64>().unwrap();
        prop_assert!(values.iter().all(|value| *value == 0.0));
    }
}
