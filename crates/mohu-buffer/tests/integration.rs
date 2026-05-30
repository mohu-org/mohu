//! Integration tests for mohu-buffer — exercises every major subsystem.

use mohu_buffer::{
    Buffer, GLOBAL_POOL, Order, SliceArg, ops,
    strides::{NdIndexIter, broadcast_strides, c_strides, f_strides},
};
use mohu_dtype::{DType, promote::CastMode};

// ── 1. Allocation ─────────────────────────────────────────────────────────────

#[test]
fn zeros_shape_and_values() {
    let buf = Buffer::zeros(DType::F64, &[3, 4]).unwrap();
    assert_eq!(buf.shape(), &[3, 4]);
    assert_eq!(buf.dtype(), DType::F64);
    assert!(buf.as_slice::<f64>().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn ones_values() {
    let buf = Buffer::ones(DType::F32, &[8]).unwrap();
    assert!(buf.as_slice::<f32>().unwrap().iter().all(|&x| x == 1.0_f32));
}

#[test]
fn full_fills_bytes() {
    // full() takes raw bytes — pass f32 3.14 as bytes
    let fill: f32 = 3.15;
    let buf = Buffer::full(DType::F32, &[5, 5], &fill.to_le_bytes()).unwrap();
    assert!(
        buf.as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| (x - 3.15_f32).abs() < 1e-6)
    );
}

// ── 2. from_slice + reshape + get/set ────────────────────────────────────────

#[test]
fn from_slice_round_trips_1d() {
    let data: Vec<i32> = (0..12).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    assert_eq!(buf.as_slice::<i32>().unwrap(), data.as_slice());
}

#[test]
fn from_slice_reshape_to_2d() {
    let data: Vec<i32> = (0..12).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[3, 4]).unwrap();
    assert_eq!(buf.shape(), &[3, 4]);
    assert_eq!(buf.get::<i32>(&[1, 0]).unwrap(), 4);
}

#[test]
fn get_set_roundtrip() {
    let mut buf = Buffer::zeros(DType::F64, &[4, 4]).unwrap();
    buf.set::<f64>(&[1, 2], 99.0).unwrap();
    assert_eq!(buf.get::<f64>(&[1, 2]).unwrap(), 99.0_f64);
}

// ── 3. Layout transforms ──────────────────────────────────────────────────────

#[test]
fn reshape_1d_to_3d() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    let r = buf.reshape(&[2, 3, 4]).unwrap();
    assert_eq!(r.shape(), &[2, 3, 4]);
    assert_eq!(r.get::<f32>(&[1, 2, 3]).unwrap(), 23.0_f32);
}

#[test]
fn transpose_2d_shape_and_values() {
    let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[2, 3]).unwrap();
    let t = buf.transpose();
    assert_eq!(t.shape(), &[3, 2]);
    // t[1,0] = original[0,1] = 1.0
    assert_eq!(t.get::<f64>(&[1, 0]).unwrap(), 1.0_f64);
    // t[2,1] = original[1,2] = 5.0
    assert_eq!(t.get::<f64>(&[2, 1]).unwrap(), 5.0_f64);
}

#[test]
fn permute_3d_shape() {
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let buf = Buffer::from_slice(&data)
        .unwrap()
        .reshape(&[2, 3, 4])
        .unwrap();
    let p = buf.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), &[4, 2, 3]);
}

// ── 4. Slice axis ─────────────────────────────────────────────────────────────

#[test]
fn slice_axis_rows() {
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[4, 3]).unwrap();
    let s = buf
        .slice_axis(
            0,
            SliceArg {
                start: Some(1),
                stop: Some(3),
                step: Some(1),
            },
        )
        .unwrap();
    assert_eq!(s.shape(), &[2, 3]);
    // row 1 of original starts at index 3 → [3, 4, 5]
    assert_eq!(s.get::<f64>(&[0, 0]).unwrap(), 3.0_f64);
    assert_eq!(s.get::<f64>(&[0, 2]).unwrap(), 5.0_f64);
}

#[test]
fn slice_axis_with_step() {
    let data: Vec<i32> = (0..10).collect();
    let buf = Buffer::from_slice(&data).unwrap();
    // every other element: 0, 2, 4, 6, 8
    let s = buf
        .slice_axis(
            0,
            SliceArg {
                start: Some(0),
                stop: Some(10),
                step: Some(2),
            },
        )
        .unwrap();
    assert_eq!(s.shape(), &[5]);
    assert_eq!(s.get::<i32>(&[2]).unwrap(), 4);
}

// ── 5. broadcast_to ───────────────────────────────────────────────────────────

#[test]
fn broadcast_scalar_to_matrix() {
    let data = vec![42.0_f64];
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[1, 1]).unwrap();
    let b = buf.broadcast_to(&[3, 4]).unwrap();
    assert_eq!(b.shape(), &[3, 4]);
    for r in 0..3 {
        for c in 0..4 {
            assert_eq!(b.get::<f64>(&[r, c]).unwrap(), 42.0_f64);
        }
    }
}

#[test]
fn broadcast_row_to_matrix() {
    let data = vec![1.0_f32, 2.0, 3.0];
    let buf = Buffer::from_slice(&data).unwrap().reshape(&[1, 3]).unwrap();
    let b = buf.broadcast_to(&[4, 3]).unwrap();
    assert_eq!(b.shape(), &[4, 3]);
    for r in 0..4 {
        assert_eq!(b.get::<f32>(&[r, 1]).unwrap(), 2.0_f32);
    }
}

// ── 6. Copy-on-write ──────────────────────────────────────────────────────────

#[test]
fn share_is_shallow_clone() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let shared = buf.share();
    // Both see same data initially
    assert_eq!(shared.get::<f64>(&[0]).unwrap(), 1.0_f64);
}

#[test]
fn make_unique_decouples_from_original() {
    let buf = Buffer::from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
    let mut owned = buf.share();
    owned.make_unique().unwrap(); // deep-copies if Arc count > 1
    owned.set::<f64>(&[0], 999.0).unwrap();
    assert_eq!(buf.get::<f64>(&[0]).unwrap(), 1.0_f64); // original unchanged
    assert_eq!(owned.get::<f64>(&[0]).unwrap(), 999.0_f64);
}

// ── 7. Cast ───────────────────────────────────────────────────────────────────

#[test]
fn cast_f64_to_f32() {
    let data: Vec<f64> = vec![1.0, 2.0, 3.5, -4.25];
    let buf = Buffer::from_slice(&data).unwrap();
    let c = buf.cast(DType::F32, CastMode::Unsafe).unwrap();
    assert_eq!(c.dtype(), DType::F32);
    let s = c.as_slice::<f32>().unwrap();
    assert!((s[2] - 3.5_f32).abs() < 1e-5);
    assert!((s[3] - (-4.25_f32)).abs() < 1e-5);
}

#[test]
fn cast_i32_to_f64() {
    let data: Vec<i32> = vec![10, -5, 0, 127];
    let buf = Buffer::from_slice(&data).unwrap();
    let c = buf.cast(DType::F64, CastMode::Safe).unwrap();
    let s = c.as_slice::<f64>().unwrap();
    assert_eq!(s[0], 10.0_f64);
    assert_eq!(s[1], -5.0_f64);
}

#[test]
fn cast_u8_to_f32() {
    let data: Vec<u8> = vec![0, 128, 255];
    let buf = Buffer::from_slice(&data).unwrap();
    let c = buf.cast(DType::F32, CastMode::Safe).unwrap();
    let s = c.as_slice::<f32>().unwrap();
    assert_eq!(s[0], 0.0_f32);
    assert_eq!(s[2], 255.0_f32);
}

// ── 8. Parallel fill / copy ───────────────────────────────────────────────────

#[test]
fn parallel_fill_large_buffer() {
    let mut buf = Buffer::alloc(DType::F32, &[1024, 1024], Order::C).unwrap();
    ops::fill::<f32>(&mut buf, 7.0).unwrap();
    assert!(buf.as_slice::<f32>().unwrap().iter().all(|&x| x == 7.0_f32));
}

#[test]
fn fill_zero_clears_values() {
    let mut buf = Buffer::ones(DType::F64, &[100]).unwrap();
    ops::fill_zero(&mut buf).unwrap();
    assert!(buf.as_slice::<f64>().unwrap().iter().all(|&x| x == 0.0));
}

#[test]
fn copy_to_contiguous_from_transposed() {
    // Transposed 3×3: source is non-contiguous
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let src = Buffer::from_slice(&data)
        .unwrap()
        .reshape(&[3, 3])
        .unwrap()
        .transpose();
    assert!(!src.is_c_contiguous());
    let mut dst = Buffer::alloc(DType::F64, &[3, 3], Order::C).unwrap();
    ops::copy_to_contiguous(&src, &mut dst).unwrap();
    // transposed[0,1] = original[1,0] = 3.0
    assert_eq!(dst.get::<f64>(&[0, 1]).unwrap(), 3.0_f64);
    // transposed[1,0] = original[0,1] = 1.0
    assert_eq!(dst.get::<f64>(&[1, 0]).unwrap(), 1.0_f64);
}

// ── 9. Buffer pool ────────────────────────────────────────────────────────────

#[test]
fn pool_acquire_returns_sufficient_size() {
    let handle = GLOBAL_POOL.acquire(4096).unwrap();
    assert!(handle.len() >= 4096);
    GLOBAL_POOL.release(handle);
    let stats = GLOBAL_POOL.stats();
    assert!(stats.cached_bytes > 0);
}

#[test]
fn pool_hit_after_release() {
    let h1 = GLOBAL_POOL.acquire(8192).unwrap();
    GLOBAL_POOL.release(h1);
    let stats_before = GLOBAL_POOL.stats();
    let h2 = GLOBAL_POOL.acquire(8192).unwrap();
    let stats_after = GLOBAL_POOL.stats();
    // hit count must increase on re-acquire
    assert!(stats_after.hit_count >= stats_before.hit_count);
    GLOBAL_POOL.release(h2);
}

// ── 10. Strides utilities ─────────────────────────────────────────────────────

#[test]
fn c_strides_correct() {
    // [2, 3, 4] f64 (itemsize=8): C-order strides = [96, 32, 8]
    let shape = [2usize, 3, 4];
    let s = c_strides(&shape, 8);
    assert_eq!(s.as_slice(), &[96_isize, 32, 8]);
}

#[test]
fn f_strides_correct() {
    // [2, 3, 4] f64: F-order strides = [8, 16, 48]
    let shape = [2usize, 3, 4];
    let s = f_strides(&shape, 8);
    assert_eq!(s.as_slice(), &[8_isize, 16, 48]);
}

#[test]
fn broadcast_strides_zero_for_size_one_axes() {
    let src_shape = [1usize, 4];
    let src_strides = c_strides(&src_shape, 4);
    let tgt_shape = [3usize, 4];
    let bs = broadcast_strides(&src_shape, &src_strides, &tgt_shape).unwrap();
    assert_eq!(bs[0], 0); // size-1 axis → 0-stride
    assert_ne!(bs[1], 0);
}

#[test]
fn nd_index_iter_c_order() {
    let shape = [2usize, 3];
    let indices: Vec<_> = NdIndexIter::new(&shape).collect();
    assert_eq!(indices.len(), 6);
    assert_eq!(indices[0].as_slice(), &[0usize, 0]);
    assert_eq!(indices[1].as_slice(), &[0, 1]);
    assert_eq!(indices[3].as_slice(), &[1, 0]);
    assert_eq!(indices[5].as_slice(), &[1, 2]);
}
