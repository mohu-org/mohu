use mohu_array::NdArray;
use mohu_dtype::DType;

#[test]
fn zeros_metadata() {
    let a = NdArray::<f64>::zeros(&[3, 4]).unwrap();
    assert_eq!(a.shape(), &[3, 4]);
    assert_eq!(a.ndim(), 2);
    assert_eq!(a.len(), 12);
    assert!(!a.is_empty());
    assert_eq!(a.dtype(), DType::F64);
}

#[test]
fn ones_metadata() {
    let a = NdArray::<f32>::ones(&[2, 2, 2]).unwrap();
    assert_eq!(a.shape(), &[2, 2, 2]);
    assert_eq!(a.len(), 8);
    assert_eq!(a.dtype(), DType::F32);
}

#[test]
fn from_slice_is_one_dimensional() {
    let a = NdArray::<i32>::from_slice(&[1, 2, 3]).unwrap();
    assert_eq!(a.shape(), &[3]);
    assert_eq!(a.ndim(), 1);
    assert_eq!(a.len(), 3);
    assert_eq!(a.dtype(), DType::I32);
}

#[test]
fn scalar_and_zero_sized_shapes() {
    let scalar = NdArray::<f64>::zeros(&[]).unwrap();
    let scalar_shape: &[usize] = &[];
    assert_eq!(scalar.shape(), scalar_shape);
    assert_eq!(scalar.ndim(), 0);
    assert_eq!(scalar.len(), 1);
    assert!(!scalar.is_empty());

    let empty = NdArray::<f64>::ones(&[2, 0, 4]).unwrap();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
}

#[test]
fn all_supported_dtypes_report_runtime_type() {
    macro_rules! check { ($($ty:ty => $dtype:expr),+ $(,)?) => { $(assert_eq!(NdArray::<$ty>::zeros(&[1]).unwrap().dtype(), $dtype);)+ }; }
    check!(
        bool => DType::Bool,
        i8 => DType::I8, i16 => DType::I16, i32 => DType::I32, i64 => DType::I64,
        u8 => DType::U8, u16 => DType::U16, u32 => DType::U32, u64 => DType::U64,
        half::f16 => DType::F16, half::bf16 => DType::BF16,
        f32 => DType::F32, f64 => DType::F64,
        num_complex::Complex<f32> => DType::C64,
        num_complex::Complex<f64> => DType::C128,
    );
}
