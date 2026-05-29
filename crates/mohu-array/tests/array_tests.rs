//! Integration tests for `NdArray` construction, shape access, and dtype.

use mohu_array::NdArray;

// ── zeros ──────────────────────────────────────────────────────────────────────

#[test]
fn zeros_f64_3x4() {
    let a = NdArray::<f64>::zeros(&[3, 4]).unwrap();
    assert_eq!(a.shape(), &[3, 4]);
    assert_eq!(a.ndim(), 2);
    assert_eq!(a.len(), 12);
    assert!(!a.is_empty());
    assert_eq!(a.dtype(), mohu_dtype::DType::F64);
}

#[test]
fn zeros_f32_1d() {
    let a = NdArray::<f32>::zeros(&[5]).unwrap();
    assert_eq!(a.shape(), &[5]);
    assert_eq!(a.ndim(), 1);
    assert_eq!(a.len(), 5);
}

#[test]
fn zeros_empty_shape() {
    let a = NdArray::<i32>::zeros(&[]).unwrap();
    let empty: &[usize] = &[];
    assert_eq!(a.shape(), empty);
    assert_eq!(a.ndim(), 0);
    assert_eq!(a.len(), 1);
    assert!(!a.is_empty());
}

// ── ones ───────────────────────────────────────────────────────────────────────

#[test]
fn ones_f64_2x3() {
    let a = NdArray::<f64>::ones(&[2, 3]).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.ndim(), 2);
    assert_eq!(a.len(), 6);
}

#[test]
fn ones_f32_from_slice_compare() {
    let a = NdArray::<f32>::ones(&[4]).unwrap();
    assert_eq!(a.shape(), &[4]);
    assert_eq!(a.len(), 4);
}

// ── from_slice ─────────────────────────────────────────────────────────────────

#[test]
fn from_slice_f64() {
    let data = [1.0, 2.0, 3.0];
    let a = NdArray::<f64>::from_slice(&data).unwrap();
    assert_eq!(a.shape(), &[3]);
    assert_eq!(a.len(), 3);
    assert_eq!(a.ndim(), 1);
    assert_eq!(a.dtype(), mohu_dtype::DType::F64);
}

#[test]
fn from_slice_f32() {
    let data = [1.0_f32, 2.0, 3.0, 4.0];
    let a = NdArray::<f32>::from_slice(&data).unwrap();
    assert_eq!(a.len(), 4);
    assert_eq!(a.shape(), &[4]);
}

#[test]
fn from_slice_i32() {
    let data = [10, 20, 30];
    let a = NdArray::<i32>::from_slice(&data).unwrap();
    assert_eq!(a.len(), 3);
    assert_eq!(a.dtype(), mohu_dtype::DType::I32);
}

#[test]
fn from_slice_empty() {
    let data: [f64; 0] = [];
    let a = NdArray::<f64>::from_slice(&data).unwrap();
    assert_eq!(a.len(), 0);
    assert!(a.is_empty());
    assert_eq!(a.shape(), &[0]);
    assert_eq!(a.ndim(), 1);
}

// ── dtype ──────────────────────────────────────────────────────────────────────

#[test]
fn dtype_for_each_numeric_type() {
    assert_eq!(NdArray::<f64>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::F64);
    assert_eq!(NdArray::<f32>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::F32);
    assert_eq!(NdArray::<i32>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::I32);
    assert_eq!(NdArray::<i64>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::I64);
    assert_eq!(NdArray::<u8>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::U8);
    assert_eq!(NdArray::<u16>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::U16);
    assert_eq!(NdArray::<u32>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::U32);
    assert_eq!(NdArray::<u64>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::U64);
    assert_eq!(NdArray::<i8>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::I8);
    assert_eq!(NdArray::<i16>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::I16);
    assert_eq!(NdArray::<bool>::zeros(&[1]).unwrap().dtype(), mohu_dtype::DType::Bool);
}

// ── edge cases ─────────────────────────────────────────────────────────────────

#[test]
fn zero_length_1d() {
    let a = NdArray::<f64>::from_slice(&[]).unwrap();
    assert!(a.is_empty());
    assert_eq!(a.len(), 0);
    assert_eq!(a.ndim(), 1);
    assert_eq!(a.shape(), &[0]);
}

#[test]
fn large_1d_ones() {
    let a = NdArray::<f64>::ones(&[1000]).unwrap();
    assert_eq!(a.len(), 1000);
    assert_eq!(a.ndim(), 1);
}
