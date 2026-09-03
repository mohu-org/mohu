//! The README quickstart, kept here so it is compiled and run by CI.
//!
//! Run with: `cargo run --example quickstart`

use mohu_buffer::{Buffer, Order};
use mohu_dtype::{CastMode, DType};

fn main() -> mohu_buffer::MohuResult<()> {
    // Build a 2x3 buffer from a typed slice. dtype is inferred from T.
    let a = Buffer::from_slice_2d::<f32>(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?;
    println!("a: dtype={} shape={:?}", a.dtype(), a.shape());

    // Views are zero-copy: transpose only rewrites the stride vector.
    let t = a.transpose();
    println!("a.T: shape={:?} strides={:?}", t.shape(), t.strides());
    assert_eq!(t.get::<f32>(&[2, 1])?, 6.0);

    // Constructors mirror NumPy.
    let z = Buffer::zeros(DType::F64, &[3, 4])?;
    let r = Buffer::arange(0.0, 10.0, 2.0, DType::I32)?;
    println!("zeros: {:?}, arange: {:?}", z.shape(), r.to_vec::<i32>()?);

    // Casting is explicit and checked against a promotion policy.
    let as_f64 = a.cast(DType::F64, CastMode::Safe)?;
    println!("cast f32 -> {}", as_f64.dtype());

    // Broadcasting produces a zero-copy view with 0-strides.
    let row = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0])?;
    let bcast = row.broadcast_to(&[2, 3])?;
    println!(
        "broadcast: shape={:?} strides={:?}",
        bcast.shape(),
        bcast.strides()
    );

    // Fortran-order allocation is a first-class option, not an afterthought.
    let f = Buffer::alloc(DType::F32, &[2, 3], Order::F)?;
    println!("F-order contiguous: {}", f.is_f_contiguous());

    Ok(())
}
