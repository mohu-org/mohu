use mohu_dtype::DType;

#[test]
fn default_dtype_is_f64() {
    assert_eq!(DType::default(), DType::F64);
}

#[test]
fn struct_deriving_default_uses_f64() {
    #[derive(Default)]
    struct Config {
        dtype: DType,
    }
    assert_eq!(Config::default().dtype, DType::F64);
}

#[test]
fn subset_iterators_match_exact_dtype_order() {
    use mohu_dtype::DType;
    assert_eq!(
        DType::iter_floats().collect::<Vec<_>>(),
        vec![DType::F16, DType::BF16, DType::F32, DType::F64]
    );
    assert_eq!(
        DType::iter_integers().collect::<Vec<_>>(),
        vec![
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64
        ]
    );
    assert_eq!(
        DType::iter_complex().collect::<Vec<_>>(),
        vec![DType::C64, DType::C128]
    );
    assert_eq!(
        DType::iter_signed().collect::<Vec<_>>(),
        vec![DType::I8, DType::I16, DType::I32, DType::I64]
    );
    assert_eq!(
        DType::iter_unsigned().collect::<Vec<_>>(),
        vec![DType::U8, DType::U16, DType::U32, DType::U64]
    );
}

#[test]
fn subset_iterators_cover_numeric_types_without_bool() {
    use mohu_dtype::DType;
    use std::collections::HashSet;
    let mut seen = HashSet::new();
    seen.extend(DType::iter_floats());
    seen.extend(DType::iter_integers());
    seen.extend(DType::iter_complex());
    seen.insert(DType::Bool);
    assert_eq!(seen.len(), 15);
    assert!(DType::iter_floats().all(DType::is_float));
    assert!(DType::iter_integers().all(DType::is_integer));
    assert!(DType::iter_complex().all(DType::is_complex));
    assert!(DType::iter_signed().all(DType::is_signed_integer));
    assert!(DType::iter_unsigned().all(DType::is_unsigned_integer));
}

#[test]
fn subset_iterators_have_expected_order_and_predicates() {
    use mohu_dtype::DType;
    assert_eq!(
        DType::iter_floats().collect::<Vec<_>>(),
        vec![DType::F16, DType::BF16, DType::F32, DType::F64]
    );
    assert_eq!(
        DType::iter_integers().collect::<Vec<_>>(),
        vec![
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64
        ]
    );
    assert_eq!(
        DType::iter_complex().collect::<Vec<_>>(),
        vec![DType::C64, DType::C128]
    );
    assert_eq!(
        DType::iter_signed().collect::<Vec<_>>(),
        vec![DType::I8, DType::I16, DType::I32, DType::I64]
    );
    assert_eq!(
        DType::iter_unsigned().collect::<Vec<_>>(),
        vec![DType::U8, DType::U16, DType::U32, DType::U64]
    );
    assert!(DType::iter_floats().all(DType::is_float));
    assert!(DType::iter_integers().all(DType::is_integer));
    assert!(DType::iter_complex().all(DType::is_complex));
    assert!(DType::iter_signed().all(DType::is_signed_integer));
    assert!(DType::iter_unsigned().all(DType::is_unsigned_integer));
}
