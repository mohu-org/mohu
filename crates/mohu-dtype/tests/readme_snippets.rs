//! Compile-checks the snippets in this crate's README.md so they cannot rot.

use mohu_dtype::{CastMode, DType, can_cast, dispatch_dtype, promote};

#[test]
fn readme_promotion_snippet() {
    assert_eq!(promote(DType::I32, DType::F32), DType::F64);
    assert!(can_cast(DType::I32, DType::F64, CastMode::Safe));
    assert!(!can_cast(DType::F64, DType::I32, CastMode::Safe));
}

macro_rules! size_of_ty {
    ($t:ty) => {
        std::mem::size_of::<$t>()
    };
}

fn size_of_dtype(dt: DType) -> usize {
    dispatch_dtype!(dt, size_of_ty)
}

#[test]
fn readme_dispatch_snippet() {
    assert_eq!(size_of_dtype(DType::F64), 8);
    assert_eq!(size_of_dtype(DType::C128), 16);
}
