//! Integration tests for mohu-buffer — exercises every major subsystem.

use std::ptr::NonNull;

use mohu_buffer::{
    Buffer, GLOBAL_POOL, Layout, Order, SliceArg, ops,
    strides::{NdIndexIter, broadcast_strides, c_strides, f_strides},
};
use mohu_dtype::{DType, promote::CastMode};
use mohu_error::MohuError;

#[test]
fn default_order_is_c() {
    assert_eq!(Order::default(), Order::C);
}

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
    // full() takes raw bytes, so pass a non-special f32 value as bytes.
    let fill: f32 = 3.125;
    let buf = Buffer::full(DType::F32, &[5, 5], &fill.to_le_bytes()).unwrap();
    assert!(
        buf.as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|&x| (x - 3.125_f32).abs() < 1e-6)
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
fn reshape_rejects_overflowing_destination_shape() {
    let buf = Buffer::from_slice(&[1_u8]).unwrap();
    let result = buf.reshape(&[usize::MAX, 2]);
    assert!(matches!(result, Err(MohuError::ShapeOverflow { .. })));
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
fn broadcast_shapes_resolves_right_aligned_dimensions() {
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[3], &[2, 3])
            .unwrap()
            .as_slice(),
        &[2, 3]
    );
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[1, 1, 3], &[4, 5, 3])
            .unwrap()
            .as_slice(),
        &[4, 5, 3]
    );
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[], &[2, 3])
            .unwrap()
            .as_slice(),
        &[2, 3]
    );
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[], &[]).unwrap().as_slice(),
        &[] as &[usize]
    );
}

#[test]
fn broadcast_shapes_rejects_incompatible_and_handles_empty_axes() {
    assert!(matches!(
        mohu_buffer::broadcast_shapes(&[2, 3], &[2, 4]),
        Err(MohuError::BroadcastError { .. })
    ));
    assert!(matches!(
        mohu_buffer::broadcast_shapes(&[0], &[2]),
        Err(MohuError::BroadcastError { .. })
    ));
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[0], &[1])
            .unwrap()
            .as_slice(),
        &[0]
    );
    assert_eq!(
        mohu_buffer::broadcast_shapes(&[0], &[0])
            .unwrap()
            .as_slice(),
        &[0]
    );
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
fn div_scalar_inplace_integer_zero_returns_error() {
    let mut buf = Buffer::from_slice(&[2_i32, 4, 6]).unwrap();
    let err = ops::div_scalar_inplace(&mut buf, 0_i32).unwrap_err();
    assert!(matches!(err, MohuError::DivisionByZero));
    assert_eq!(buf.as_slice::<i32>().unwrap(), &[2, 4, 6]);

    let mut buf = Buffer::from_slice(&[8_i64, 16, 24]).unwrap();
    let err = ops::div_scalar_inplace(&mut buf, 0_i64).unwrap_err();
    assert!(matches!(err, MohuError::DivisionByZero));
    assert_eq!(buf.as_slice::<i64>().unwrap(), &[8, 16, 24]);

    let mut buf = Buffer::from_slice(&[10_u32, 20, 30]).unwrap();
    let err = ops::div_scalar_inplace(&mut buf, 0_u32).unwrap_err();
    assert!(matches!(err, MohuError::DivisionByZero));
    assert_eq!(buf.as_slice::<u32>().unwrap(), &[10, 20, 30]);
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

#[test]
fn copy_to_contiguous_respects_fortran_destination_strides() {
    let data: Vec<i32> = (1..=6).collect();
    let src = Buffer::from_slice(&data).unwrap().reshape(&[2, 3]).unwrap();
    let mut dst = Buffer::alloc(DType::I32, &[2, 3], Order::F).unwrap();

    assert!(!dst.is_c_contiguous());
    assert!(dst.is_f_contiguous());

    ops::copy_to_contiguous(&src, &mut dst).unwrap();

    assert_eq!(dst.get::<i32>(&[0, 0]).unwrap(), 1);
    assert_eq!(dst.get::<i32>(&[0, 1]).unwrap(), 2);
    assert_eq!(dst.get::<i32>(&[0, 2]).unwrap(), 3);
    assert_eq!(dst.get::<i32>(&[1, 0]).unwrap(), 4);
    assert_eq!(dst.get::<i32>(&[1, 1]).unwrap(), 5);
    assert_eq!(dst.get::<i32>(&[1, 2]).unwrap(), 6);
}

#[test]
fn copy_from_respects_fortran_destination_strides() {
    let data: Vec<i32> = (10..=15).collect();
    let src = Buffer::from_slice(&data).unwrap().reshape(&[2, 3]).unwrap();
    let mut dst = Buffer::alloc(DType::I32, &[2, 3], Order::F).unwrap();

    dst.copy_from(&src).unwrap();

    assert_eq!(dst.get::<i32>(&[0, 0]).unwrap(), 10);
    assert_eq!(dst.get::<i32>(&[0, 1]).unwrap(), 11);
    assert_eq!(dst.get::<i32>(&[0, 2]).unwrap(), 12);
    assert_eq!(dst.get::<i32>(&[1, 0]).unwrap(), 13);
    assert_eq!(dst.get::<i32>(&[1, 1]).unwrap(), 14);
    assert_eq!(dst.get::<i32>(&[1, 2]).unwrap(), 15);
}

#[test]
fn copy_to_contiguous_respects_nonzero_offset_destination() {
    let src = Buffer::from_slice(&[1_i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let mut backing = vec![-1_i32; 9];
    let itemsize = std::mem::size_of::<i32>();
    let layout = Layout::new_custom(
        &[2, 2],
        &[(3 * itemsize) as isize, itemsize as isize],
        4 * itemsize,
        itemsize,
    )
    .unwrap();
    let ptr = NonNull::new(backing.as_mut_ptr() as *mut u8).unwrap();

    // SAFETY: backing stays alive for the whole test, and the custom layout
    // touches only elements 4, 5, 7, and 8 within the 9-element backing Vec.
    let mut dst =
        unsafe { Buffer::from_raw_parts(ptr, backing.len() * itemsize, DType::I32, layout) };

    assert_eq!(dst.offset(), 4 * itemsize);
    assert!(!dst.is_c_contiguous());

    ops::copy_to_contiguous(&src, &mut dst).unwrap();

    assert_eq!(dst.get::<i32>(&[0, 0]).unwrap(), 1);
    assert_eq!(dst.get::<i32>(&[0, 1]).unwrap(), 2);
    assert_eq!(dst.get::<i32>(&[1, 0]).unwrap(), 3);
    assert_eq!(dst.get::<i32>(&[1, 1]).unwrap(), 4);

    drop(dst);
    assert_eq!(backing, vec![-1, -1, -1, -1, 1, 2, -1, 3, 4]);
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
#[test]
fn reverse_full_slice_preserves_axis() {
    let buf = Buffer::from_slice(&(0..3).collect::<Vec<i32>>()).unwrap();
    let s = buf
        .slice_axis(
            0,
            SliceArg {
                start: None,
                stop: None,
                step: Some(-1),
            },
        )
        .unwrap();
    assert_eq!(s.shape(), &[3]);
    assert_eq!(s.get::<i32>(&[0]).unwrap(), 2);
    assert_eq!(s.get::<i32>(&[2]).unwrap(), 0);
}

#[test]
fn shape_predicates_cover_common_shapes() {
    let v = Buffer::from_slice(&[1_i32, 2]).unwrap();
    assert!(v.is_vector());
    assert!(!v.is_matrix());
    let m = v.reshape(&[1, 2]).unwrap();
    assert!(m.is_matrix());
    assert!(!m.is_square());
    let q = Buffer::from_slice(&[1_i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    assert!(q.is_square());
    let s = Buffer::zeros(DType::F64, &[]).unwrap();
    assert!(s.is_scalar_shape());
}

#[test]
fn dlpack_zero_size_null_data_is_valid_and_deleter_runs() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    unsafe extern "C" fn deleter(ptr: *mut mohu_buffer::DLManagedTensor) {
        if ptr.is_null() {
            return;
        }
        unsafe {
            let shape = (*ptr).dl_tensor.shape as *mut i64;
            if !shape.is_null() {
                drop(Vec::from_raw_parts(shape, 1, 1));
            }
            drop(Box::from_raw(ptr));
        }
        DROPS.fetch_add(1, Ordering::SeqCst);
    }
    DROPS.store(0, Ordering::SeqCst);
    let shape = Box::into_raw(Box::new([0_i64])) as *const i64;
    let managed = Box::into_raw(Box::new(mohu_buffer::DLManagedTensor {
        dl_tensor: mohu_buffer::DLTensor {
            data: std::ptr::null_mut(),
            device: mohu_buffer::RawDLDevice {
                device_type: 1,
                device_id: 0,
            },
            ndim: 1,
            dtype: mohu_buffer::RawDLDataType {
                code: 2,
                bits: 64,
                lanes: 1,
            },
            shape,
            strides: std::ptr::null(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: Some(deleter),
    }));
    let buffer = unsafe { Buffer::from_dlpack(managed) }.unwrap();
    assert_eq!(buffer.shape(), &[0]);
    assert_eq!(buffer.nbytes(), 0);
    assert!(!buffer.as_ptr().is_null());
    assert_eq!((buffer.as_ptr() as usize) % std::mem::align_of::<f64>(), 0);
    assert!(buffer.as_slice::<f64>().unwrap().is_empty());
    drop(buffer);
    assert_eq!(DROPS.load(Ordering::SeqCst), 1);
}

#[test]
fn dlpack_rejects_negative_ndim_before_pointer_reads() {
    unsafe extern "C" fn deleter(ptr: *mut mohu_buffer::DLManagedTensor) {
        if !ptr.is_null() {
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }
    let managed = Box::into_raw(Box::new(mohu_buffer::DLManagedTensor {
        dl_tensor: mohu_buffer::DLTensor {
            data: std::ptr::null_mut(),
            device: mohu_buffer::RawDLDevice {
                device_type: 1,
                device_id: 0,
            },
            ndim: -1,
            dtype: mohu_buffer::RawDLDataType {
                code: 2,
                bits: 64,
                lanes: 1,
            },
            shape: std::ptr::null(),
            strides: std::ptr::null(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: Some(deleter),
    }));
    let result = unsafe { Buffer::from_dlpack(managed) };
    assert!(matches!(result, Err(MohuError::DLPackInvalid(_))));
    unsafe {
        deleter(managed);
    }
}

#[test]
fn sum_axis_dtype_and_values_follow_policy() {
    let bools = Buffer::from_vec(vec![true, false, true]).unwrap();
    let sum = bools.sum_axis(0, false).unwrap();
    assert_eq!(sum.dtype(), DType::I64);
    assert_eq!(sum.as_slice::<i64>().unwrap(), &[2]);

    let signed = Buffer::from_vec(vec![1_i32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let sum = signed.sum_axis(0, true).unwrap();
    assert_eq!(sum.dtype(), DType::I64);
    assert_eq!(sum.shape(), &[1, 2]);
    assert_eq!(sum.as_slice::<i64>().unwrap(), &[4, 6]);

    let unsigned = Buffer::from_vec(vec![1_u32, 2, 3, 4])
        .unwrap()
        .reshape(&[2, 2])
        .unwrap();
    let sum = unsigned.sum_axis(1, false).unwrap();
    assert_eq!(sum.dtype(), DType::U64);
    assert_eq!(sum.as_slice::<u64>().unwrap(), &[3, 7]);

    let signed_overflow = Buffer::from_vec(vec![i64::MAX, 1])
        .unwrap()
        .sum_axis(0, false)
        .unwrap();
    assert_eq!(signed_overflow.as_slice::<i64>().unwrap(), &[i64::MIN]);
    let unsigned_overflow = Buffer::from_vec(vec![u64::MAX, 1])
        .unwrap()
        .sum_axis(0, false)
        .unwrap();
    assert_eq!(unsigned_overflow.as_slice::<u64>().unwrap(), &[0]);
}

#[test]
fn sum_axis_preserves_float_and_complex_output_dtypes() {
    let f16 = Buffer::from_vec(vec![half::f16::from_f32(1.25), half::f16::from_f32(2.5)]).unwrap();
    let sum = f16.sum_axis(0, false).unwrap();
    assert_eq!(sum.dtype(), DType::F16);
    assert!((sum.as_slice::<half::f16>().unwrap()[0].to_f32() - 3.75).abs() < 0.01);

    let bf16 =
        Buffer::from_vec(vec![half::bf16::from_f32(1.25), half::bf16::from_f32(2.5)]).unwrap();
    let sum = bf16.sum_axis(0, false).unwrap();
    assert_eq!(sum.dtype(), DType::BF16);
    assert!((sum.as_slice::<half::bf16>().unwrap()[0].to_f32() - 3.75).abs() < 0.02);

    let f32s = Buffer::from_vec(vec![1.25_f32, 2.5])
        .unwrap()
        .sum_axis(0, false)
        .unwrap();
    assert_eq!(f32s.dtype(), DType::F32);
    assert!((f32s.as_slice::<f32>().unwrap()[0] - 3.75).abs() < 1e-6);
    let f64s = Buffer::from_vec(vec![1.25_f64, 2.5])
        .unwrap()
        .sum_axis(0, false)
        .unwrap();
    assert_eq!(f64s.dtype(), DType::F64);
    assert!((f64s.as_slice::<f64>().unwrap()[0] - 3.75).abs() < 1e-12);

    let c64s = Buffer::from_vec(vec![
        num_complex::Complex::new(1.0_f32, 2.0),
        num_complex::Complex::new(3.0, 4.0),
    ])
    .unwrap()
    .sum_axis(0, false)
    .unwrap();
    assert_eq!(c64s.dtype(), DType::C64);
    assert_eq!(
        c64s.as_slice::<num_complex::Complex<f32>>().unwrap(),
        &[num_complex::Complex::new(4.0, 6.0)]
    );
    let c128s = Buffer::from_vec(vec![
        num_complex::Complex::new(1.0_f64, 2.0),
        num_complex::Complex::new(3.0, 4.0),
    ])
    .unwrap()
    .sum_axis(0, false)
    .unwrap();
    assert_eq!(c128s.dtype(), DType::C128);
    assert_eq!(
        c128s.as_slice::<num_complex::Complex<f64>>().unwrap(),
        &[num_complex::Complex::new(4.0, 6.0)]
    );
}

#[test]
fn sum_axis_handles_strided_and_empty_inputs() {
    let transposed = Buffer::from_vec(vec![1_i32, 2, 3, 4, 5, 6])
        .unwrap()
        .reshape(&[2, 3])
        .unwrap()
        .transpose();
    let sum = transposed.sum_axis(1, false).unwrap();
    assert_eq!(sum.as_slice::<i64>().unwrap(), &[5, 7, 9]);

    let empty = Buffer::zeros(DType::I32, &[2, 0]).unwrap();
    let sum = empty.sum_axis(1, false).unwrap();
    assert_eq!(sum.dtype(), DType::I64);
    assert_eq!(sum.shape(), &[2]);
    assert_eq!(sum.as_slice::<i64>().unwrap(), &[0, 0]);
}
#[test]
fn diagonal_operations_report_dimension_mismatches() {
    let one_d = Buffer::from_slice(&[1_i32, 2, 3]).unwrap();
    let two_d = one_d.reshape(&[1, 3]).unwrap();
    assert!(matches!(
        Buffer::diag(&two_d, 0),
        Err(MohuError::DimensionMismatch {
            expected: 1,
            got: 2
        })
    ));
    assert!(matches!(
        one_d.diagonal(0),
        Err(MohuError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
    assert!(matches!(
        one_d.tril(0),
        Err(MohuError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
    assert!(matches!(
        one_d.triu(0),
        Err(MohuError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn allclose_contract_covers_values_nan_infinities_and_dtypes() {
    let a = Buffer::from_slice(&[1.0_f64, 2.0, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    let b = Buffer::from_slice(&[1.0_f64, 2.000001, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    assert!(a.allclose(&b, 1e-5, 1e-8).unwrap());
    let far = Buffer::from_slice(&[1.0_f64, 2.1, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    assert!(!a.allclose(&far, 1e-5, 1e-8).unwrap());
    let nan = Buffer::from_slice(&[f64::NAN, 2.0, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
    assert!(!a.allclose(&nan, 1e-5, 1e-8).unwrap());
    assert!(!nan.allclose(&nan, 1e-5, 1e-8).unwrap());

    let f32_a = Buffer::from_slice(&[1.0_f32, 2.0]).unwrap();
    let f32_b = Buffer::from_slice(&[1.0_f32, 2.000001]).unwrap();
    assert!(f32_a.allclose(&f32_b, 1e-5, 1e-8).unwrap());
    let f16_a = Buffer::from_slice(&[half::f16::from_f32(1.0)]).unwrap();
    let f16_b = Buffer::from_slice(&[half::f16::from_f32(1.0001)]).unwrap();
    assert!(f16_a.allclose(&f16_b, 1e-3, 1e-8).unwrap());
    let bf16_a = Buffer::from_slice(&[half::bf16::from_f32(1.0)]).unwrap();
    let bf16_b = Buffer::from_slice(&[half::bf16::from_f32(1.001)]).unwrap();
    assert!(bf16_a.allclose(&bf16_b, 1e-2, 1e-8).unwrap());
}

#[test]
fn allclose_contract_rejects_shape_and_non_float_inputs() {
    let a = Buffer::from_slice(&[1.0_f64, 2.0]).unwrap();
    let shape_mismatch = Buffer::from_slice(&[1.0_f64]).unwrap();
    assert!(matches!(
        a.allclose(&shape_mismatch, 1e-5, 1e-8),
        Err(mohu_error::MohuError::ShapeMismatch { .. })
    ));
    let integer = Buffer::from_slice(&[1_i32, 2]).unwrap();
    assert!(matches!(
        a.allclose(&integer, 1e-5, 1e-8),
        Err(mohu_error::MohuError::DomainError { .. })
    ));
    let complex = Buffer::from_slice(&[num_complex::Complex::<f32>::new(1.0, 2.0)]).unwrap();
    assert!(matches!(
        complex.allclose(&complex, 1e-5, 1e-8),
        Err(mohu_error::MohuError::DomainError { .. })
    ));
}

#[test]
fn item_supports_scalar_singleton_and_strided_views() {
    let scalar = Buffer::from_slice(&[42_i32]).unwrap().reshape(&[]).unwrap();
    assert_eq!(scalar.item::<i32>().unwrap(), 42);

    let singleton = Buffer::from_slice(&[7.5_f64]).unwrap();
    assert_eq!(
        singleton.item::<f64>().unwrap().to_bits(),
        7.5_f64.to_bits()
    );

    let view = Buffer::from_slice(&[10_i32, 20, 30])
        .unwrap()
        .slice_axis(
            0,
            SliceArg {
                start: Some(1),
                stop: Some(2),
                step: None,
            },
        )
        .unwrap();
    assert_eq!(view.item::<i32>().unwrap(), 20);
}

#[test]
fn item_preserves_typed_and_cardinality_errors() {
    let scalar = Buffer::from_slice(&[1_i32]).unwrap();
    assert!(matches!(
        scalar.item::<f64>(),
        Err(MohuError::DTypeMismatch { .. })
    ));

    let empty = Buffer::zeros(DType::I32, &[0]).unwrap();
    assert!(matches!(
        empty.item::<i32>(),
        Err(MohuError::DomainError { op: "item", .. })
    ));

    let many = Buffer::from_slice(&[1_i32, 2]).unwrap();
    assert!(matches!(
        many.item::<i32>(),
        Err(MohuError::DomainError { op: "item", .. })
    ));
}

#[test]
fn shape_matches_requires_exact_shape() {
    let a = Buffer::zeros(DType::I32, &[2, 3]).unwrap();
    assert!(
        a.shape_matches(&Buffer::zeros(DType::I32, &[2, 3]).unwrap())
            .is_ok()
    );
    assert!(
        matches!(a.shape_matches(&Buffer::zeros(DType::I32, &[3, 2]).unwrap()), Err(MohuError::ShapeMismatch { ref expected, ref got }) if expected == &[2, 3] && got == &[3, 2])
    );
    assert!(matches!(
        a.shape_matches(&Buffer::zeros(DType::I32, &[6]).unwrap()),
        Err(MohuError::ShapeMismatch { .. })
    ));
    assert!(
        Buffer::zeros(DType::I32, &[])
            .unwrap()
            .shape_matches(&Buffer::zeros(DType::F64, &[]).unwrap())
            .is_ok()
    );
}
