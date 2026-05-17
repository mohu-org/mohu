use mohu_dtype::{DType, ALL_DTYPES};
use mohu_testing::strategies::{
    arb_dtype, arb_float_dtype, arb_integer_dtype, arb_standard_float_dtype,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn arb_dtype_is_known_variant(dtype in arb_dtype()) {
        prop_assert!(ALL_DTYPES.contains(&dtype));
    }

    #[test]
    fn arb_float_dtype_yields_float(dtype in arb_float_dtype()) {
        prop_assert!(dtype.is_float());
    }

    #[test]
    fn arb_integer_dtype_yields_integer(dtype in arb_integer_dtype()) {
        prop_assert!(dtype.is_integer());
    }

    #[test]
    fn arb_standard_float_dtype_yields_f32_or_f64(dtype in arb_standard_float_dtype()) {
        prop_assert!(matches!(dtype, DType::F32 | DType::F64));
    }
}
