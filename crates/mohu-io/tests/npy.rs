use std::collections::HashMap;

use mohu_core::{mohu_buffer::Buffer, mohu_dtype::DType};
use mohu_io::npy::{load_npy, load_npz, save_npy, save_npz};
use tempfile::tempdir;

#[test]
fn npy_roundtrip_f64_vector() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("vec.npy");
    let original = Buffer::from_slice(&[1.0_f64, 2.0, 3.5]).expect("buffer");

    save_npy(&path, &original).expect("save");
    let loaded = load_npy(&path).expect("load");

    assert_eq!(loaded.dtype(), DType::F64);
    assert_eq!(loaded.shape(), &[3]);
    assert_eq!(loaded.as_slice::<f64>().expect("slice"), &[1.0, 2.0, 3.5]);
}

#[test]
fn npz_roundtrip_named_arrays() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("arrays.npz");
    let a = Buffer::from_slice(&[1.0_f64, 2.0]).expect("a");
    let b = Buffer::from_slice(&[3_i32, 4, 5]).expect("b");

    save_npz(&path, &[("weights", &a), ("indices", &b)]).expect("save");
    let loaded: HashMap<String, Buffer> = load_npz(&path).expect("load");

    assert_eq!(loaded.len(), 2);
    let weights = loaded.get("weights").expect("weights");
    assert_eq!(weights.as_slice::<f64>().expect("f64"), &[1.0, 2.0]);
    let indices = loaded.get("indices").expect("indices");
    assert_eq!(indices.as_slice::<i32>().expect("i32"), &[3, 4, 5]);
}
