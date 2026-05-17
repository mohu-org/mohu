//! Proptest strategies for mohu types.

use mohu_dtype::{DType, ALL_DTYPES};
use proptest::prelude::*;

const FLOAT_DTYPES: &[DType] = &[DType::F16, DType::BF16, DType::F32, DType::F64];
const INTEGER_DTYPES: &[DType] = &[
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
];
const STANDARD_FLOAT_DTYPES: &[DType] = &[DType::F32, DType::F64];

/// Arbitrary [`DType`] from all supported variants.
pub fn arb_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(ALL_DTYPES.as_slice())
}

/// Arbitrary floating-point [`DType`] (F16, BF16, F32, F64).
pub fn arb_float_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(FLOAT_DTYPES)
}

/// Arbitrary integer [`DType`] (signed and unsigned, I8–U64).
pub fn arb_integer_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(INTEGER_DTYPES)
}

/// Arbitrary standard IEEE float [`DType`] (F32 or F64 only).
pub fn arb_standard_float_dtype() -> impl Strategy<Value = DType> {
    prop::sample::select(STANDARD_FLOAT_DTYPES)
}
