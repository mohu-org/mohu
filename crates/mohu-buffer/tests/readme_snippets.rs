//! Compile-checks the snippets and stated constants in this crate's README.md
//! (and the workspace README quickstart) so they cannot rot.

use mohu_buffer::{Buffer, MMAP_THRESHOLD, Order, SIMD_ALIGN};
use mohu_dtype::{CastMode, DType};

#[test]
fn readme_usage_snippet() -> mohu_buffer::MohuResult<()> {
    let a = Buffer::from_slice_2d::<f32>(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?;

    let t = a.transpose();
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.get::<f32>(&[2, 1])?, 6.0);

    let r = Buffer::arange(0.0, 10.0, 2.0, DType::I32)?;
    assert_eq!(r.to_vec::<i32>()?, vec![0, 2, 4, 6, 8]);

    let row = Buffer::from_slice::<f32>(&[10.0, 20.0, 30.0])?;
    assert_eq!(row.broadcast_to(&[2, 3])?.strides(), &[0, 4]);

    assert!(Buffer::alloc(DType::F32, &[2, 3], Order::F)?.is_f_contiguous());

    // Also exercised by the workspace README quickstart.
    let z = Buffer::zeros(DType::F64, &[3, 4])?;
    assert_eq!(z.shape(), &[3, 4]);
    assert_eq!(a.cast(DType::F64, CastMode::Safe)?.dtype(), DType::F64);

    Ok(())
}

#[test]
fn readme_stated_constants() {
    assert_eq!(SIMD_ALIGN, 64, "README documents 64-byte SIMD alignment");
    assert_eq!(
        MMAP_THRESHOLD,
        1024 * 1024,
        "README documents a 1 MiB mmap threshold"
    );
}

#[test]
fn readme_sharing_semantics() -> mohu_buffer::MohuResult<()> {
    let a = Buffer::from_slice::<f32>(&[1.0, 2.0, 3.0])?;
    let b = a.share();
    assert!(a.is_shared(), "share() must not deep-copy");
    drop(b);
    Ok(())
}
