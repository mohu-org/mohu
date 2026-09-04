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
