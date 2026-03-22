// buffer_basics.rs — Buffer creation, element access, slicing, and reshaping
//
// Demonstrates mohu's Buffer type: creating arrays from various sources,
// reading and writing elements, and performing zero-copy view operations
// like transpose, reshape, slice, and broadcast.
//
// NumPy equivalents:
//   a = np.array([1, 2, 3], dtype=np.float32)
//   a = np.zeros((3, 4), dtype=np.float64)
//   a = np.arange(0, 10, 1, dtype=np.int32)
//   a.T, a.reshape(2, 6), a[::2], np.broadcast_to(a, (3, 4))

use mohu_buffer::{
    Buffer, SliceArg,
    strides::{c_strides, f_strides, NdIndexIter, unravel_index, ravel_multi_index},
};
use mohu_dtype::DType;

fn main() {
    // ── Buffer creation from a Rust slice ──────────────────────────────────
    // NumPy: a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    let a = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    println!("from_slice: dtype={}, shape={:?}, len={}", a.dtype(), a.shape(), a.len());

    // Read elements back
    let v: f32 = a.get(&[2]).unwrap();
    println!("  a[2] = {v}"); // 3.0

    // ── 2D buffer from nested slices ───────────────────────────────────────
    // NumPy: a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    let rows: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]];
    let a2d = Buffer::from_slice_2d(rows).unwrap();
    println!("\nfrom_slice_2d: shape={:?}, strides={:?}", a2d.shape(), a2d.strides());
    let v: f64 = a2d.get(&[1, 2]).unwrap();
    println!("  a[1,2] = {v}"); // 6.0

    // ── Zeros, ones, full ──────────────────────────────────────────────────
    // NumPy: np.zeros((3, 4), dtype=np.float64)
    let z = Buffer::zeros(DType::F64, &[3, 4]).unwrap();
    println!("\nzeros: shape={:?}, dtype={}", z.shape(), z.dtype());
    let v: f64 = z.get(&[0, 0]).unwrap();
    println!("  z[0,0] = {v}"); // 0.0

    // NumPy: np.ones((2, 3), dtype=np.float32)
    let o = Buffer::ones(DType::F32, &[2, 3]).unwrap();
    let v: f32 = o.get(&[1, 1]).unwrap();
    println!("ones[1,1] = {v}"); // 1.0

    // NumPy: np.full((2, 2), 42.0, dtype=np.float64)
    // Buffer::full takes raw bytes; 42.0_f64 is 8 bytes
    let fill_val: f64 = 42.0;
    let f = Buffer::full(DType::F64, &[2, 2], &fill_val.to_ne_bytes()).unwrap();
    let v: f64 = f.get(&[0, 1]).unwrap();
    println!("full[0,1] = {v}"); // 42.0

    // ── arange and linspace ────────────────────────────────────────────────
    // NumPy: np.arange(0, 10, 2, dtype=np.int32)
    let ar = Buffer::arange(0.0, 10.0, 2.0, DType::I32).unwrap();
    let data = ar.to_vec::<i32>().unwrap();
    println!("\narange(0, 10, 2): {:?}", data); // [0, 2, 4, 6, 8]

    // NumPy: np.linspace(0, 1, 5, dtype=np.float64)
    let ls = Buffer::linspace(0.0, 1.0, 5, true, DType::F64).unwrap();
    let data = ls.to_vec::<f64>().unwrap();
    println!("linspace(0, 1, 5): {:?}", data); // [0.0, 0.25, 0.5, 0.75, 1.0]

    // ── eye and diag ───────────────────────────────────────────────────────
    // NumPy: np.eye(3, dtype=np.float64)
    let eye = Buffer::eye(3, 3, 0, DType::F64).unwrap();
    println!("\neye(3,3):");
    for i in 0..3 {
        for j in 0..3 {
            let v: f64 = eye.get(&[i, j]).unwrap();
            print!(" {v:.0}");
        }
        println!();
    }

    // ── Transpose (zero-copy view operation) ───────────────────────────────
    // NumPy: a.T
    let t = a2d.transpose();
    println!("\nTranspose: shape={:?}, strides={:?}", t.shape(), t.strides());
    let v: f64 = t.get(&[2, 1]).unwrap(); // was a2d[1,2] = 6.0
    println!("  a.T[2,1] = {v}"); // 6.0

    // ── Reshape (zero-copy for contiguous) ─────────────────────────────────
    // NumPy: a.reshape(3, 2)
    let r = a2d.reshape(&[3, 2]).unwrap();
    println!("\nReshape(3,2): shape={:?}", r.shape());
    let v: f64 = r.get(&[2, 1]).unwrap();
    println!("  reshaped[2,1] = {v}"); // 6.0

    // ── Slicing (zero-copy view with adjusted strides) ─────────────────────
    // NumPy: a[::2]  (every other element)
    let five = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
    let sliced = five.slice_axis(0, SliceArg { start: Some(0), stop: None, step: Some(2) }).unwrap();
    let data = sliced.to_vec::<f32>().unwrap();
    println!("\nslice [::2]: {:?}", data); // [10.0, 30.0, 50.0]

    // ── Broadcast (zero-copy virtual replication) ──────────────────────────
    // NumPy: np.broadcast_to(np.array([1, 2, 3]), (3, 3))
    let row = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0]).unwrap()
        .expand_dims(0).unwrap();
    let bc = row.broadcast_to(&[3, 3]).unwrap();
    println!("\nBroadcast to (3,3): shape={:?}", bc.shape());
    for i in 0..3 {
        for j in 0..3 {
            let v: f32 = bc.get(&[i, j]).unwrap();
            print!(" {v:.0}");
        }
        println!();
    }

    // ── Layout properties ──────────────────────────────────────────────────
    println!("\n── Layout inspection ──");
    println!("a2d C-contiguous: {}", a2d.is_c_contiguous());
    println!("a2d F-contiguous: {}", a2d.is_f_contiguous());
    println!("a2d.T C-contiguous: {}", t.is_c_contiguous());  // false after transpose
    println!("a2d.T F-contiguous: {}", t.is_f_contiguous());  // true

    // ── Stride arithmetic ──────────────────────────────────────────────────
    println!("\n── Stride arithmetic ──");
    let shape = [3, 4];
    let c_str = c_strides(&shape, 4); // itemsize=4 (f32)
    let f_str = f_strides(&shape, 4);
    println!("C-strides for (3,4) f32: {:?}", c_str.as_slice()); // [16, 4]
    println!("F-strides for (3,4) f32: {:?}", f_str.as_slice()); // [4, 12]

    // ── N-dimensional index iteration ──────────────────────────────────────
    println!("\n── NdIndexIter ──");
    let iter = NdIndexIter::new(&[2, 3]);
    let indices: Vec<_> = iter.collect();
    println!("Indices for (2,3):");
    for idx in &indices {
        print!("  {:?}", idx.as_slice());
    }
    println!();

    // ── Index conversion ───────────────────────────────────────────────────
    // NumPy: np.unravel_index(5, (2, 3)) => (1, 2)
    let idx = unravel_index(5, &[2, 3]).unwrap();
    println!("\nunravel_index(5, (2,3)) = {:?}", idx.as_slice());

    // NumPy: np.ravel_multi_index((1, 2), (2, 3)) => 5
    let flat = ravel_multi_index(&[1, 2], &[2, 3]).unwrap();
    println!("ravel_multi_index((1,2), (2,3)) = {flat}");

    // ── Describe ───────────────────────────────────────────────────────────
    println!("\n{}", a2d.describe());

    println!("\nDone!");
}
