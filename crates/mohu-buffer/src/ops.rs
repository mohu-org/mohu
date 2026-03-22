/// Buffer-level operations: fill, copy, cast, and parallel element transforms.
///
/// All operations that touch large arrays use Rayon's `par_iter` for
/// automatic multi-core parallelism with no GIL and no explicit threading.
///
/// # Dispatch
///
/// Every runtime-dtype operation uses `dispatch_dtype!` from `mohu_dtype`
/// to monomorphise a generic kernel into 15 concrete specialisations.
/// The compiler inlines and optimises each specialisation individually,
/// enabling auto-vectorisation at the call site.
use rayon::prelude::*;

use mohu_dtype::{
    cast::cast_scalar_unchecked,
    dispatch_dtype,
    promote::{can_cast, CastMode},
    scalar::Scalar,
};
use mohu_error::{MohuError, MohuResult};

use crate::{
    buffer::Buffer,
    strides::StridedByteIter,
};

// ─── fill_raw ────────────────────────────────────────────────────────────────

/// Fills every element of `buf` by repeating the raw bytes in `fill_bytes`.
///
/// `fill_bytes.len()` must equal `buf.itemsize()`.
///
/// For contiguous buffers, uses a fast Rayon chunk-parallel fill.
/// For non-contiguous buffers, falls back to stride iteration.
pub fn fill_raw(buf: &mut Buffer, fill_bytes: &[u8]) -> MohuResult<()> {
    let itemsize = buf.itemsize();
    if fill_bytes.len() != itemsize {
        return Err(MohuError::bug(format!(
            "fill_raw: fill_bytes.len()={} != itemsize={}",
            fill_bytes.len(), itemsize
        )));
    }
    if !buf.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    buf.make_unique()?;

    if buf.is_c_contiguous() {
        let total = buf.len() * itemsize;
        // SAFETY: buf is uniquely owned, C-contiguous, pointer is valid.
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buf.as_mut_ptr(), total)
        };
        // Parallel fill: split into 4KiB chunks and fill each on Rayon threads.
        let chunk_size = 4096.max(itemsize * 64);
        slice.par_chunks_mut(chunk_size).for_each(|chunk| {
            let mut pos = 0;
            while pos + itemsize <= chunk.len() {
                chunk[pos..pos + itemsize].copy_from_slice(fill_bytes);
                pos += itemsize;
            }
        });
    } else {
        // Non-contiguous: walk strides.
        let raw_ptr = unsafe { buf.as_mut_ptr() };
        for off in StridedByteIter::new(buf.shape(), buf.strides(), buf.offset()) {
            // SAFETY: stride iterator yields valid byte offsets within the buffer.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    fill_bytes.as_ptr(),
                    raw_ptr.add(off),
                    itemsize,
                );
            }
        }
    }
    Ok(())
}

// ─── fill ────────────────────────────────────────────────────────────────────

/// Fills every element of `buf` with `value`.
///
/// `T::DTYPE` must match `buf.dtype()`.
pub fn fill<T>(buf: &mut Buffer, value: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync,
{
    if T::DTYPE != buf.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: T::DTYPE.to_string(),
            got:      buf.dtype().to_string(),
        });
    }
    if !buf.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    buf.make_unique()?;

    if buf.is_c_contiguous() {
        let len = buf.len();
        let ptr = unsafe { buf.as_mut_ptr() as *mut T };
        // SAFETY: uniqueness checked, C-contiguous, len is correct.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        slice.par_iter_mut().for_each(|x| *x = value);
    } else {
        let raw_ptr = unsafe { buf.as_mut_ptr() };
        for off in StridedByteIter::new(buf.shape(), buf.strides(), buf.offset()) {
            unsafe {
                let ptr = raw_ptr.add(off) as *mut T;
                ptr.write_unaligned(value);
            }
        }
    }
    Ok(())
}

// ─── fill_zero / fill_one ─────────────────────────────────────────────────────

/// Fills every byte of the buffer's element range with zero.
pub fn fill_zero(buf: &mut Buffer) -> MohuResult<()> {
    if !buf.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    buf.make_unique()?;
    if buf.is_c_contiguous() {
        let total = buf.len() * buf.itemsize();
        unsafe {
            buf.as_mut_ptr().write_bytes(0, total);
        }
    } else {
        let fill = vec![0u8; buf.itemsize()];
        fill_raw(buf, &fill)?;
    }
    Ok(())
}

/// Fills every element with its type's "one" value.
///
/// Dispatches at runtime over all 15 dtypes.
pub fn fill_one(buf: &mut Buffer) -> MohuResult<()> {
    macro_rules! do_fill_one {
        ($T:ty) => {{
            fill(buf, <$T as Scalar>::ONE)
        }};
    }
    dispatch_dtype!(buf.dtype(), do_fill_one)
}

// ─── copy_to_contiguous ───────────────────────────────────────────────────────

/// Copies every element of `src` into `dst` in C order.
///
/// Works with any combination of contiguous and non-contiguous source/dst.
/// For the fully-contiguous case, uses a Rayon parallel `memcpy`.
pub fn copy_to_contiguous(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    if src.dtype() != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: src.dtype().to_string(),
            got:      dst.dtype().to_string(),
        });
    }
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if !dst.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    dst.make_unique()?;

    let itemsize = src.itemsize();

    if src.is_c_contiguous() && dst.is_c_contiguous() {
        // Parallel memcpy: split into cache-friendly 64 KiB chunks.
        let src_bytes = src.len() * itemsize;
        let chunk     = 65536_usize.max(itemsize);
        unsafe {
            let s = std::slice::from_raw_parts(src.as_ptr(), src_bytes);
            let d = std::slice::from_raw_parts_mut(dst.as_mut_ptr(), src_bytes);
            s.par_chunks(chunk)
                .zip(d.par_chunks_mut(chunk))
                .for_each(|(sc, dc)| dc.copy_from_slice(sc));
        }
    } else {
        // General strided → contiguous copy.
        let src_raw = src.as_ptr();
        let dst_raw = unsafe { dst.as_mut_ptr() };
        let dst_itemsize = dst.itemsize();

        for (elem_idx, src_off) in StridedByteIter::new(
            src.shape(), src.strides(), src.offset(),
        ).enumerate() {
            let dst_off = elem_idx * dst_itemsize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_raw.add(src_off),
                    dst_raw.add(dst_off),
                    itemsize,
                );
            }
        }
    }
    Ok(())
}

// ─── cast_copy ───────────────────────────────────────────────────────────────

/// Casts every element of `src` into `dst` under `mode`, in parallel.
///
/// Uses nested `dispatch_dtype!` to monomorphise over all 225 (src, dst) type
/// pairs with zero virtual dispatch.  For the identity cast (same dtype),
/// delegates to `copy_to_contiguous`.
pub fn cast_copy(src: &Buffer, dst: &mut Buffer, mode: CastMode) -> MohuResult<()> {
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if src.dtype() == dst.dtype() {
        return copy_to_contiguous(src, dst);
    }
    if !can_cast(src.dtype(), dst.dtype(), mode) {
        return Err(MohuError::InvalidCast {
            from:   src.dtype().to_string(),
            to:     dst.dtype().to_string(),
            reason: format!("{mode:?} cast is not permitted"),
        });
    }
    if !dst.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    dst.make_unique()?;

    let src_dtype = src.dtype();
    let dst_dtype = dst.dtype();

    macro_rules! cast_src {
        ($S:ty) => {{
            macro_rules! cast_dst {
                ($D:ty) => {{
                    cast_typed::<$S, $D>(src, dst)
                }};
            }
            dispatch_dtype!(dst_dtype, cast_dst)
        }};
    }
    dispatch_dtype!(src_dtype, cast_src)
}

fn cast_typed<S: Scalar + Send + Sync, D: Scalar + Send + Sync>(
    src: &Buffer,
    dst: &mut Buffer,
) -> MohuResult<()> {
    let len = src.len();

    if src.is_c_contiguous() && dst.is_c_contiguous() {
        // Fast path: parallel cast over flat slices.
        let src_ptr = src.as_ptr() as *const S;
        let dst_ptr = unsafe { dst.as_mut_ptr() } as *mut D;

        // SAFETY: both are C-contiguous, len is verified equal above.
        let src_slice = unsafe { std::slice::from_raw_parts(src_ptr, len) };
        let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst_ptr, len) };

        src_slice
            .par_iter()
            .zip(dst_slice.par_iter_mut())
            .for_each(|(s, d)| {
                *d = cast_scalar_unchecked::<S, D>(*s);
            });
    } else {
        // Slow path: stride iteration.
        let src_raw = src.as_ptr();
        let dst_raw = unsafe { dst.as_mut_ptr() };
        let dst_itemsize = D::ITEMSIZE;

        for (elem_idx, src_off) in StridedByteIter::new(
            src.shape(), src.strides(), src.offset(),
        ).enumerate() {
            let dst_off = elem_idx * dst_itemsize;
            unsafe {
                let s = (src_raw.add(src_off) as *const S).read_unaligned();
                let d = cast_scalar_unchecked::<S, D>(s);
                (dst_raw.add(dst_off) as *mut D).write_unaligned(d);
            }
        }
    }
    Ok(())
}

// ─── parallel_map ─────────────────────────────────────────────────────────────

/// Applies `f` to every `S`-typed element of `src`, writing `D`-typed results
/// to `dst`.  Both must be C-contiguous.
pub fn parallel_map<S, D, F>(src: &Buffer, dst: &mut Buffer, f: F) -> MohuResult<()>
where
    S: Scalar + Send + Sync,
    D: Scalar + Send + Sync,
    F: Fn(S) -> D + Send + Sync,
{
    if !src.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if !dst.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    dst.make_unique()?;

    let len = src.len();
    let s_ptr = src.as_ptr() as *const S;
    let d_ptr = unsafe { dst.as_mut_ptr() } as *mut D;
    let src_slice = unsafe { std::slice::from_raw_parts(s_ptr, len) };
    let dst_slice = unsafe { std::slice::from_raw_parts_mut(d_ptr, len) };

    src_slice
        .par_iter()
        .zip(dst_slice.par_iter_mut())
        .for_each(|(s, d)| *d = f(*s));

    Ok(())
}

// ─── parallel_inplace ─────────────────────────────────────────────────────────

/// Applies `f` in-place to every element of `buf`.  Requires C-contiguous.
pub fn parallel_inplace<T, F>(buf: &mut Buffer, f: F) -> MohuResult<()>
where
    T: Scalar + Send + Sync,
    F: Fn(T) -> T + Send + Sync,
{
    if !buf.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if !buf.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    buf.make_unique()?;

    let len = buf.len();
    let ptr = unsafe { buf.as_mut_ptr() } as *mut T;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    slice.par_iter_mut().for_each(|x| *x = f(*x));
    Ok(())
}

// ─── reduce ──────────────────────────────────────────────────────────────────

/// Parallel reduction over a C-contiguous buffer.
///
/// Computes `init` combined with every element using `combine`.
/// Uses Rayon's `map_reduce` to parallelise across chunks.
pub fn reduce<T, R, F, G>(
    buf:     &Buffer,
    init:    R,
    map_fn:  F,
    combine: G,
) -> MohuResult<R>
where
    T: Scalar + Send + Sync,
    R: Clone + Send + Sync,
    F: Fn(T) -> R + Send + Sync,
    G: Fn(R, R) -> R + Send + Sync,
{
    if T::DTYPE != buf.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: T::DTYPE.to_string(),
            got:      buf.dtype().to_string(),
        });
    }
    if !buf.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    let len = buf.len();
    let ptr = buf.as_ptr() as *const T;
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

    Ok(slice
        .par_iter()
        .map(|&x| map_fn(x))
        .reduce(|| init.clone(), &combine))
}

// ─── fill_sequential ─────────────────────────────────────────────────────────

/// Fills `buf` with an arithmetic sequence: `start, start+step, start+2*step, …`
///
/// Equivalent to `np.arange` applied in-place.  Uses Rayon to compute each
/// element independently in parallel (no dependency between elements).
pub fn fill_sequential<T>(buf: &mut Buffer, start: T, step: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    if T::DTYPE != buf.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: T::DTYPE.to_string(),
            got:      buf.dtype().to_string(),
        });
    }
    if !buf.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    buf.make_unique()?;

    if buf.is_c_contiguous() {
        let len = buf.len();
        let ptr = unsafe { buf.as_mut_ptr() as *mut T };
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        // Each element is independent: index-based computation, no data race.
        slice.par_iter_mut().enumerate().for_each(|(i, v)| {
            // start + step * i  — computed without shared mutable state
            let mut acc = start;
            for _ in 0..i { acc = acc + step; } // naive but correct; compiler may vectorise
            *v = acc;
        });
    } else {
        // Non-contiguous: sequential stride walk (rare path).
        let raw_ptr = unsafe { buf.as_mut_ptr() };
        for (i, off) in StridedByteIter::new(buf.shape(), buf.strides(), buf.offset()).enumerate() {
            let mut acc = start;
            for _ in 0..i { acc = acc + step; }
            unsafe {
                (raw_ptr.add(off) as *mut T).write_unaligned(acc);
            }
        }
    }
    Ok(())
}

// ─── parallel_scan ────────────────────────────────────────────────────────────

/// Computes an inclusive parallel prefix scan (e.g. cumulative sum) over
/// a C-contiguous buffer, writing results into `dst`.
///
/// Uses the work-efficient Blelloch / Kogge-Stone two-pass algorithm:
///
/// 1. **Down-sweep**: each Rayon chunk computes its own local scan and records
///    its chunk total.
/// 2. **Sequential carry**: the per-chunk totals are prefix-summed sequentially
///    (cheap: only `n_chunks` elements).
/// 3. **Up-sweep**: each chunk adds its carry offset in parallel.
///
/// This achieves O(n) work and O(log n) span, identical to serial complexity.
pub fn parallel_scan<T, F>(
    src:      &Buffer,
    dst:      &mut Buffer,
    identity: T,
    f:        F,
) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync,
    F: Fn(T, T) -> T + Send + Sync + Copy,
{
    if T::DTYPE != src.dtype() || T::DTYPE != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: T::DTYPE.to_string(),
            got:      src.dtype().to_string(),
        });
    }
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if !src.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if !dst.is_writeable() {
        return Err(MohuError::ReadOnly);
    }
    dst.make_unique()?;

    let len    = src.len();
    let src_s  = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const T, len) };
    let dst_s  = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut T, len) };

    // Chunk size: balance parallelism vs. carry overhead.
    let n_threads  = rayon::current_num_threads().max(1);
    let chunk_size = (len / n_threads).max(256).next_power_of_two();
    let n_chunks   = len.div_ceil(chunk_size);

    // Step 1: local scans + collect chunk totals.
    let mut chunk_totals: Vec<T> = vec![identity; n_chunks];
    src_s
        .par_chunks(chunk_size)
        .zip(dst_s.par_chunks_mut(chunk_size))
        .zip(chunk_totals.par_iter_mut())
        .for_each(|((src_c, dst_c), total)| {
            let mut acc = identity;
            for (s, d) in src_c.iter().zip(dst_c.iter_mut()) {
                acc = f(acc, *s);
                *d = acc;
            }
            *total = acc;
        });

    // Step 2: sequential prefix sum over chunk totals.
    for i in 1..n_chunks {
        chunk_totals[i] = f(chunk_totals[i - 1], chunk_totals[i]);
    }

    // Step 3: add carry to all chunks after the first.
    if n_chunks > 1 {
        dst_s
            .par_chunks_mut(chunk_size)
            .enumerate()
            .skip(1)
            .for_each(|(ci, chunk)| {
                let carry = chunk_totals[ci - 1];
                for v in chunk.iter_mut() {
                    *v = f(carry, *v);
                }
            });
    }

    Ok(())
}

// ─── where_select ─────────────────────────────────────────────────────────────

/// Element-wise conditional selection: `dst[i] = if mask[i] != 0 { a[i] } else { b[i] }`.
///
/// `mask` must have dtype `U8` (boolean mask: 0 = false, non-zero = true).
/// `a`, `b`, and `dst` must have the same dtype and shape.
/// All four buffers must be C-contiguous.
pub fn where_select<T: Scalar + Copy + Send + Sync>(
    mask: &Buffer,
    a:    &Buffer,
    b:    &Buffer,
    dst:  &mut Buffer,
) -> MohuResult<()> {
    use mohu_dtype::DType;

    if mask.dtype() != DType::U8 {
        return Err(MohuError::DTypeMismatch {
            expected: "U8".to_string(),
            got:      mask.dtype().to_string(),
        });
    }
    for (name, buf) in [("a", a), ("b", b)] {
        if T::DTYPE != buf.dtype() {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got:      buf.dtype().to_string(),
            });
        }
        if buf.shape() != mask.shape() {
            return Err(MohuError::ShapeMismatch {
                expected: mask.shape().to_vec(),
                got:      buf.shape().to_vec(),
            });
        }
        if !buf.is_c_contiguous() {
            let _ = name;
            return Err(MohuError::NonContiguous);
        }
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    if dst.shape() != mask.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: mask.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    dst.make_unique()?;

    let len    = mask.len();
    let m_ptr  = mask.as_ptr();
    let a_ptr  = a.as_ptr() as *const T;
    let b_ptr  = b.as_ptr() as *const T;
    let d_ptr  = unsafe { dst.as_mut_ptr() } as *mut T;

    let m_s = unsafe { std::slice::from_raw_parts(m_ptr, len) };
    let a_s = unsafe { std::slice::from_raw_parts(a_ptr, len) };
    let b_s = unsafe { std::slice::from_raw_parts(b_ptr, len) };
    let d_s = unsafe { std::slice::from_raw_parts_mut(d_ptr, len) };

    m_s.par_iter()
        .zip(a_s.par_iter())
        .zip(b_s.par_iter())
        .zip(d_s.par_iter_mut())
        .for_each(|(((m, av), bv), dv)| {
            *dv = if *m != 0 { *av } else { *bv };
        });

    Ok(())
}

// ─── clip ─────────────────────────────────────────────────────────────────────

/// Clips every element of `src` to `[lo, hi]`, writing into `dst`.
///
/// Equivalent to `np.clip`.  Both buffers must be C-contiguous.
pub fn clip<T>(src: &Buffer, dst: &mut Buffer, lo: T, hi: T) -> MohuResult<()>
where
    T: Scalar + PartialOrd + Copy + Send + Sync,
{
    if T::DTYPE != src.dtype() || T::DTYPE != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: T::DTYPE.to_string(),
            got:      src.dtype().to_string(),
        });
    }
    if src.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if !src.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    dst.make_unique()?;

    let len   = src.len();
    let s_ptr = src.as_ptr() as *const T;
    let d_ptr = unsafe { dst.as_mut_ptr() } as *mut T;
    let s_s   = unsafe { std::slice::from_raw_parts(s_ptr, len) };
    let d_s   = unsafe { std::slice::from_raw_parts_mut(d_ptr, len) };

    s_s.par_iter().zip(d_s.par_iter_mut()).for_each(|(s, d)| {
        *d = if *s < lo { lo } else if *s > hi { hi } else { *s };
    });

    Ok(())
}

// ─── gather ───────────────────────────────────────────────────────────────────

/// Indexed gather: `dst[i] = src[indices[i]]`.
///
/// `indices` must have dtype `I64`.  `src` and `dst` must be 1-D and C-contiguous.
/// Panics in debug mode on out-of-bounds indices.
pub fn gather(src: &Buffer, indices: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;

    if indices.dtype() != DType::I64 {
        return Err(MohuError::DTypeMismatch {
            expected: "I64".to_string(),
            got:      indices.dtype().to_string(),
        });
    }
    if src.dtype() != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: src.dtype().to_string(),
            got:      dst.dtype().to_string(),
        });
    }
    if !src.is_c_contiguous() || !indices.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if indices.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: indices.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    dst.make_unique()?;

    let itemsize  = src.dtype().itemsize();
    let src_len   = src.len();
    let n_idx     = indices.len();

    // Build typed slices from the raw pointers.  All three buffers are
    // exclusively borrowed (src immutably, dst mutably) for this call.
    // Using &[u8] for src makes it Sync, enabling safe sharing across Rayon threads.
    let src_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(src.as_ptr(), src_len * itemsize)
    };
    let idx_s: &[i64] = unsafe {
        std::slice::from_raw_parts(indices.as_ptr() as *const i64, n_idx)
    };
    let dst_b: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(dst.as_mut_ptr(), n_idx * itemsize)
    };

    idx_s.par_iter().zip(dst_b.par_chunks_mut(itemsize)).for_each(|(&idx, out)| {
        let i = if idx < 0 {
            (src_len as i64 + idx) as usize
        } else {
            idx as usize
        };
        debug_assert!(i < src_len, "gather: index {i} out of bounds (len {src_len})");
        let i = i.min(src_len.saturating_sub(1)); // clamp in release
        out.copy_from_slice(&src_bytes[i * itemsize..(i + 1) * itemsize]);
    });

    Ok(())
}

// ─── scatter ──────────────────────────────────────────────────────────────────

/// Indexed scatter: `dst[indices[i]] = src[i]`.
///
/// `indices` must have dtype `I64`.  `src` and `dst` must be 1-D and C-contiguous.
///
/// **Warning**: if two indices are equal, the result is non-deterministic (a
/// data race between Rayon threads).  For safe scatter with duplicates, use
/// the sequential variant or ensure indices are unique.
pub fn scatter(dst: &mut Buffer, indices: &Buffer, src: &Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;

    if indices.dtype() != DType::I64 {
        return Err(MohuError::DTypeMismatch {
            expected: "I64".to_string(),
            got:      indices.dtype().to_string(),
        });
    }
    if src.dtype() != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: src.dtype().to_string(),
            got:      dst.dtype().to_string(),
        });
    }
    if !src.is_c_contiguous() || !indices.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if indices.len() != src.len() {
        return Err(MohuError::ShapeMismatch {
            expected: indices.shape().to_vec(),
            got:      src.shape().to_vec(),
        });
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    dst.make_unique()?;

    let itemsize  = src.dtype().itemsize();
    let dst_len   = dst.len();
    let n_src     = src.len();
    let idx_ptr   = indices.as_ptr() as *const i64;
    let src_ptr   = src.as_ptr();
    let dst_ptr   = unsafe { dst.as_mut_ptr() };

    let idx_s = unsafe { std::slice::from_raw_parts(idx_ptr, n_src) };
    let src_b = unsafe { std::slice::from_raw_parts(src_ptr, n_src * itemsize) };

    // Sequential scatter to avoid data races on duplicate indices.
    for (i, &idx) in idx_s.iter().enumerate() {
        let j = if idx < 0 {
            (dst_len as i64 + idx) as usize
        } else {
            idx as usize
        };
        if j < dst_len {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_b.as_ptr().add(i * itemsize),
                    dst_ptr.add(j * itemsize),
                    itemsize,
                );
            }
        }
    }

    Ok(())
}

// ─── Arithmetic scalar ops ────────────────────────────────────────────────────

/// Adds `scalar` to every element of `buf` in-place.
pub fn add_scalar_inplace<T>(buf: &mut Buffer, scalar: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync + std::ops::Add<Output = T>,
{
    parallel_inplace::<T, _>(buf, |x| x + scalar)
}

/// Subtracts `scalar` from every element of `buf` in-place.
pub fn sub_scalar_inplace<T>(buf: &mut Buffer, scalar: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync + std::ops::Sub<Output = T>,
{
    parallel_inplace::<T, _>(buf, |x| x - scalar)
}

/// Multiplies every element of `buf` by `scalar` in-place.
pub fn mul_scalar_inplace<T>(buf: &mut Buffer, scalar: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync + std::ops::Mul<Output = T>,
{
    parallel_inplace::<T, _>(buf, |x| x * scalar)
}

/// Divides every element of `buf` by `scalar` in-place.
pub fn div_scalar_inplace<T>(buf: &mut Buffer, scalar: T) -> MohuResult<()>
where
    T: Scalar + Copy + Send + Sync + std::ops::Div<Output = T>,
{
    parallel_inplace::<T, _>(buf, |x| x / scalar)
}

// ─── Unary copy ops via dispatch ─────────────────────────────────────────────

/// Computes the absolute value of every element: `dst[i] = |src[i]|`.
///
/// For unsigned types (Bool, U8, U16, U32, U64) abs is the identity — produces
/// a byte-for-byte copy.  For signed integer and float types the absolute value
/// is computed via an f64 round-trip.  For complex types each component (re, im)
/// is made non-negative independently.
pub fn abs_copy(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;
    use num_complex::Complex;
    // Macro for types that fit through an f64 round-trip (all ints + f16/bf16/f32/f64).
    macro_rules! abs_via_f64 {
        ($T:ty) => {{
            parallel_map::<$T, $T, _>(src, dst, |x| {
                if let Some(s) = num_traits::cast::<$T, f64>(x) {
                    num_traits::cast::<f64, $T>(s.abs()).unwrap_or(x)
                } else {
                    x
                }
            })
        }};
    }
    match src.dtype() {
        // Unsigned and bool: abs is the identity.
        DType::Bool | DType::U8 | DType::U16 | DType::U32 | DType::U64 => copy_to_contiguous(src, dst),
        // Signed integers and reals: round-trip through f64.
        DType::I8   => abs_via_f64!(i8),
        DType::I16  => abs_via_f64!(i16),
        DType::I32  => abs_via_f64!(i32),
        DType::I64  => abs_via_f64!(i64),
        DType::F16  => abs_via_f64!(::half::f16),
        DType::BF16 => abs_via_f64!(::half::bf16),
        DType::F32  => parallel_map::<f32, f32, _>(src, dst, |x| x.abs()),
        DType::F64  => parallel_map::<f64, f64, _>(src, dst, |x| x.abs()),
        // Complex: abs each component independently (preserves dtype).
        DType::C64  => parallel_map::<Complex<f32>, Complex<f32>, _>(src, dst, |x| {
            Complex::new(x.re.abs(), x.im.abs())
        }),
        DType::C128 => parallel_map::<Complex<f64>, Complex<f64>, _>(src, dst, |x| {
            Complex::new(x.re.abs(), x.im.abs())
        }),
    }
}

/// Negates every element: `dst[i] = -src[i]`.
///
/// For unsigned integer types, wraps around (two's-complement negation via i128).
/// For `Bool`, negation is undefined — produces a copy.
/// For complex types both real and imaginary parts are negated.
pub fn neg_copy(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;
    use num_complex::Complex;
    macro_rules! neg_via_i128 {
        ($T:ty) => {{
            parallel_map::<$T, $T, _>(src, dst, |x| {
                if let Some(v) = num_traits::cast::<$T, i128>(x) {
                    num_traits::cast::<i128, $T>(-v).unwrap_or(x)
                } else {
                    x
                }
            })
        }};
    }
    match src.dtype() {
        // Bool has no meaningful negation; produce a copy.
        DType::Bool => copy_to_contiguous(src, dst),
        DType::I8   => neg_via_i128!(i8),
        DType::I16  => neg_via_i128!(i16),
        DType::I32  => neg_via_i128!(i32),
        DType::I64  => neg_via_i128!(i64),
        DType::U8   => parallel_map::<u8,  u8,  _>(src, dst, |x| x.wrapping_neg()),
        DType::U16  => parallel_map::<u16, u16, _>(src, dst, |x| x.wrapping_neg()),
        DType::U32  => parallel_map::<u32, u32, _>(src, dst, |x| x.wrapping_neg()),
        DType::U64  => parallel_map::<u64, u64, _>(src, dst, |x| x.wrapping_neg()),
        DType::F16  => {
            parallel_map::<::half::f16, ::half::f16, _>(src, dst, |x| {
                ::half::f16::from_f32(-x.to_f32())
            })
        }
        DType::BF16 => {
            parallel_map::<::half::bf16, ::half::bf16, _>(src, dst, |x| {
                ::half::bf16::from_f32(-x.to_f32())
            })
        }
        DType::F32  => parallel_map::<f32, f32, _>(src, dst, |x| -x),
        DType::F64  => parallel_map::<f64, f64, _>(src, dst, |x| -x),
        DType::C64  => parallel_map::<Complex<f32>, Complex<f32>, _>(src, dst, |x| -x),
        DType::C128 => parallel_map::<Complex<f64>, Complex<f64>, _>(src, dst, |x| -x),
    }
}

/// Computes element-wise sqrt: `dst[i] = sqrt(src[i])`.
///
/// Src must have dtype F32 or F64.  Negative values produce NaN.
pub fn sqrt_copy(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;
    match src.dtype() {
        DType::F32 => parallel_map::<f32, f32, _>(src, dst, |x| x.sqrt()),
        DType::F64 => parallel_map::<f64, f64, _>(src, dst, |x| x.sqrt()),
        other => Err(MohuError::DTypeMismatch {
            expected: "F32 or F64".to_string(),
            got:      other.to_string(),
        }),
    }
}

/// Computes element-wise natural log: `dst[i] = ln(src[i])`.
pub fn ln_copy(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;
    match src.dtype() {
        DType::F32 => parallel_map::<f32, f32, _>(src, dst, |x| x.ln()),
        DType::F64 => parallel_map::<f64, f64, _>(src, dst, |x| x.ln()),
        other => Err(MohuError::DTypeMismatch {
            expected: "F32 or F64".to_string(),
            got:      other.to_string(),
        }),
    }
}

/// Computes element-wise exp: `dst[i] = e^src[i]`.
pub fn exp_copy(src: &Buffer, dst: &mut Buffer) -> MohuResult<()> {
    use mohu_dtype::DType;
    match src.dtype() {
        DType::F32 => parallel_map::<f32, f32, _>(src, dst, |x| x.exp()),
        DType::F64 => parallel_map::<f64, f64, _>(src, dst, |x| x.exp()),
        other => Err(MohuError::DTypeMismatch {
            expected: "F32 or F64".to_string(),
            got:      other.to_string(),
        }),
    }
}

// ─── Flip axis (copy) ─────────────────────────────────────────────────────────

/// Copies elements of `src` into `dst` with axis `axis` reversed.
///
/// `dst` must be C-contiguous and have the same shape as `src`.
/// `src` may be non-contiguous.
pub fn flip_axis_copy(src: &Buffer, dst: &mut Buffer, axis: usize) -> MohuResult<()> {
    if src.dtype() != dst.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: src.dtype().to_string(),
            got:      dst.dtype().to_string(),
        });
    }
    if src.shape() != dst.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got:      dst.shape().to_vec(),
        });
    }
    if axis >= src.ndim() {
        return Err(MohuError::bug(format!(
            "flip_axis_copy: axis {axis} out of bounds for ndim {}", src.ndim()
        )));
    }
    if !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    dst.make_unique()?;

    let itemsize = src.dtype().itemsize();
    let src_raw  = src.as_ptr();
    let dst_raw  = unsafe { dst.as_mut_ptr() };

    use crate::strides::NdIndexIter;

    // Walk destination in C order; compute the flipped source index.
    for (dst_flat, mut idx) in NdIndexIter::new(dst.shape()).enumerate() {
        // Flip the requested axis.
        let dim   = src.shape()[axis];
        idx[axis] = dim - 1 - idx[axis];

        let src_off = src.layout().byte_offset(idx.as_slice())
            .expect("NdIndexIter always in bounds");
        let dst_off = dst_flat * itemsize;

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_raw.add(src_off),
                dst_raw.add(dst_off),
                itemsize,
            );
        }
    }

    Ok(())
}

// ─── Parallel reduction helpers ───────────────────────────────────────────────

/// Computes the sum of all elements as f64 (works for any numeric dtype).
///
/// Non-contiguous arrays are first converted to contiguous.
pub fn sum_all_f64(buf: &Buffer) -> MohuResult<f64> {
    use mohu_dtype::DType;
    // Complex: no canonical scalar sum; error out.
    if matches!(buf.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "sum_all_f64",
            dtype: buf.dtype().to_string(),
        });
    }
    // Bool: true=1 / false=0 — reinterpret bytes as u8.
    if buf.dtype() == DType::Bool {
        let c;
        let s: &[u8] = if buf.is_c_contiguous() {
            unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) }
        } else {
            c = buf.to_contiguous()?;
            unsafe { std::slice::from_raw_parts(c.as_ptr(), c.len()) }
        };
        return Ok(s.par_iter().map(|&b| b as f64).sum());
    }
    // All other numeric types implement NumCast.
    macro_rules! do_sum {
        ($T:ty) => {{
            if buf.is_c_contiguous() {
                let s = unsafe {
                    std::slice::from_raw_parts(buf.as_ptr() as *const $T, buf.len())
                };
                let sum: f64 = s.par_iter().map(|&x| {
                    num_traits::cast::<$T, f64>(x).unwrap_or(0.0)
                }).sum();
                Ok(sum)
            } else {
                let c = buf.to_contiguous()?;
                let s = unsafe {
                    std::slice::from_raw_parts(c.as_ptr() as *const $T, c.len())
                };
                let sum: f64 = s.par_iter().map(|&x| {
                    num_traits::cast::<$T, f64>(x).unwrap_or(0.0)
                }).sum();
                Ok(sum)
            }
        }};
    }
    match buf.dtype() {
        DType::I8   => do_sum!(i8),
        DType::I16  => do_sum!(i16),
        DType::I32  => do_sum!(i32),
        DType::I64  => do_sum!(i64),
        DType::U8   => do_sum!(u8),
        DType::U16  => do_sum!(u16),
        DType::U32  => do_sum!(u32),
        DType::U64  => do_sum!(u64),
        DType::F16  => do_sum!(::half::f16),
        DType::BF16 => do_sum!(::half::bf16),
        DType::F32  => do_sum!(f32),
        DType::F64  => do_sum!(f64),
        // Bool and C64/C128 handled above.
        _ => unreachable!(),
    }
}

/// Computes the minimum element value as f64.
pub fn min_all_f64(buf: &Buffer) -> MohuResult<f64> {
    use mohu_dtype::DType;
    if matches!(buf.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "min_all_f64",
            dtype: buf.dtype().to_string(),
        });
    }
    if buf.dtype() == DType::Bool {
        if buf.is_empty() { return Ok(f64::INFINITY); }
        let c;
        let s: &[u8] = if buf.is_c_contiguous() {
            unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) }
        } else {
            c = buf.to_contiguous()?;
            unsafe { std::slice::from_raw_parts(c.as_ptr(), c.len()) }
        };
        return Ok(s.par_iter().map(|&b| b as u64).min().unwrap_or(0) as f64);
    }
    macro_rules! do_min {
        ($T:ty) => {{
            if buf.is_empty() {
                return Ok(f64::INFINITY);
            }
            let c;
            let s: &[$T] = if buf.is_c_contiguous() {
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const $T, buf.len()) }
            } else {
                c = buf.to_contiguous()?;
                unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, c.len()) }
            };
            let m = s.par_iter().cloned().reduce_with(|a, b| {
                match a.partial_cmp(&b) {
                    Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal) => a,
                    _ => b,
                }
            }).unwrap_or(s[0]);
            Ok(num_traits::cast::<$T, f64>(m).unwrap_or(f64::NAN))
        }};
    }
    match buf.dtype() {
        DType::I8   => do_min!(i8),
        DType::I16  => do_min!(i16),
        DType::I32  => do_min!(i32),
        DType::I64  => do_min!(i64),
        DType::U8   => do_min!(u8),
        DType::U16  => do_min!(u16),
        DType::U32  => do_min!(u32),
        DType::U64  => do_min!(u64),
        DType::F16  => do_min!(::half::f16),
        DType::BF16 => do_min!(::half::bf16),
        DType::F32  => do_min!(f32),
        DType::F64  => do_min!(f64),
        _ => unreachable!(),
    }
}

/// Computes the maximum element value as f64.
pub fn max_all_f64(buf: &Buffer) -> MohuResult<f64> {
    use mohu_dtype::DType;
    if matches!(buf.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "max_all_f64",
            dtype: buf.dtype().to_string(),
        });
    }
    if buf.dtype() == DType::Bool {
        if buf.is_empty() { return Ok(f64::NEG_INFINITY); }
        let c;
        let s: &[u8] = if buf.is_c_contiguous() {
            unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) }
        } else {
            c = buf.to_contiguous()?;
            unsafe { std::slice::from_raw_parts(c.as_ptr(), c.len()) }
        };
        return Ok(s.par_iter().map(|&b| b as u64).max().unwrap_or(0) as f64);
    }
    macro_rules! do_max {
        ($T:ty) => {{
            if buf.is_empty() {
                return Ok(f64::NEG_INFINITY);
            }
            let c;
            let s: &[$T] = if buf.is_c_contiguous() {
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const $T, buf.len()) }
            } else {
                c = buf.to_contiguous()?;
                unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, c.len()) }
            };
            let m = s.par_iter().cloned().reduce_with(|a, b| {
                match a.partial_cmp(&b) {
                    Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal) => a,
                    _ => b,
                }
            }).unwrap_or(s[0]);
            Ok(num_traits::cast::<$T, f64>(m).unwrap_or(f64::NAN))
        }};
    }
    match buf.dtype() {
        DType::I8   => do_max!(i8),
        DType::I16  => do_max!(i16),
        DType::I32  => do_max!(i32),
        DType::I64  => do_max!(i64),
        DType::U8   => do_max!(u8),
        DType::U16  => do_max!(u16),
        DType::U32  => do_max!(u32),
        DType::U64  => do_max!(u64),
        DType::F16  => do_max!(::half::f16),
        DType::BF16 => do_max!(::half::bf16),
        DType::F32  => do_max!(f32),
        DType::F64  => do_max!(f64),
        _ => unreachable!(),
    }
}

/// Returns the flat index of the minimum element.
pub fn argmin_flat(buf: &Buffer) -> MohuResult<usize> {
    use mohu_dtype::DType;
    if matches!(buf.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "argmin_flat",
            dtype: buf.dtype().to_string(),
        });
    }
    if buf.dtype() == DType::Bool {
        if buf.is_empty() { return Ok(0); }
        let c;
        let s: &[u8] = if buf.is_c_contiguous() {
            unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) }
        } else {
            c = buf.to_contiguous()?;
            unsafe { std::slice::from_raw_parts(c.as_ptr(), c.len()) }
        };
        return Ok(s.iter().enumerate()
            .min_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0));
    }
    macro_rules! do_argmin {
        ($T:ty) => {{
            if buf.is_empty() { return Ok(0); }
            let c;
            let s: &[$T] = if buf.is_c_contiguous() {
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const $T, buf.len()) }
            } else {
                c = buf.to_contiguous()?;
                unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, c.len()) }
            };
            Ok(s.par_iter().enumerate()
                .reduce_with(|(ai, av), (bi, bv)| {
                    if av <= bv { (ai, av) } else { (bi, bv) }
                })
                .map(|(i, _)| i)
                .unwrap_or(0))
        }};
    }
    match buf.dtype() {
        DType::I8   => do_argmin!(i8),
        DType::I16  => do_argmin!(i16),
        DType::I32  => do_argmin!(i32),
        DType::I64  => do_argmin!(i64),
        DType::U8   => do_argmin!(u8),
        DType::U16  => do_argmin!(u16),
        DType::U32  => do_argmin!(u32),
        DType::U64  => do_argmin!(u64),
        DType::F16  => do_argmin!(::half::f16),
        DType::BF16 => do_argmin!(::half::bf16),
        DType::F32  => do_argmin!(f32),
        DType::F64  => do_argmin!(f64),
        _ => unreachable!(),
    }
}

/// Returns the flat index of the maximum element.
pub fn argmax_flat(buf: &Buffer) -> MohuResult<usize> {
    use mohu_dtype::DType;
    if matches!(buf.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "argmax_flat",
            dtype: buf.dtype().to_string(),
        });
    }
    if buf.dtype() == DType::Bool {
        if buf.is_empty() { return Ok(0); }
        let c;
        let s: &[u8] = if buf.is_c_contiguous() {
            unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) }
        } else {
            c = buf.to_contiguous()?;
            unsafe { std::slice::from_raw_parts(c.as_ptr(), c.len()) }
        };
        return Ok(s.iter().enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0));
    }
    macro_rules! do_argmax {
        ($T:ty) => {{
            if buf.is_empty() { return Ok(0); }
            let c;
            let s: &[$T] = if buf.is_c_contiguous() {
                unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const $T, buf.len()) }
            } else {
                c = buf.to_contiguous()?;
                unsafe { std::slice::from_raw_parts(c.as_ptr() as *const $T, c.len()) }
            };
            Ok(s.par_iter().enumerate()
                .reduce_with(|(ai, av), (bi, bv)| {
                    if av >= bv { (ai, av) } else { (bi, bv) }
                })
                .map(|(i, _)| i)
                .unwrap_or(0))
        }};
    }
    match buf.dtype() {
        DType::I8   => do_argmax!(i8),
        DType::I16  => do_argmax!(i16),
        DType::I32  => do_argmax!(i32),
        DType::I64  => do_argmax!(i64),
        DType::U8   => do_argmax!(u8),
        DType::U16  => do_argmax!(u16),
        DType::U32  => do_argmax!(u32),
        DType::U64  => do_argmax!(u64),
        DType::F16  => do_argmax!(::half::f16),
        DType::BF16 => do_argmax!(::half::bf16),
        DType::F32  => do_argmax!(f32),
        DType::F64  => do_argmax!(f64),
        _ => unreachable!(),
    }
}

// ─── Non-temporal large fill (x86_64 AVX2) ───────────────────────────────────

/// Fills a C-contiguous F32 buffer using non-temporal (streaming) AVX2 stores.
///
/// Bypasses the CPU cache — achieves peak DRAM write bandwidth for buffers
/// > a few MiB where the data will not be immediately re-read.
/// Falls back to the standard Rayon fill on non-x86_64 platforms.
pub fn fill_nontemporal_f32_buf(buf: &mut Buffer, value: f32) -> MohuResult<()> {
    use mohu_dtype::DType;
    if buf.dtype() != DType::F32 {
        return Err(MohuError::DTypeMismatch {
            expected: "F32".to_string(),
            got:      buf.dtype().to_string(),
        });
    }
    if !buf.is_c_contiguous() { return Err(MohuError::NonContiguous); }
    if !buf.is_writeable()    { return Err(MohuError::ReadOnly); }
    buf.make_unique()?;

    let len = buf.len();
    let ptr = unsafe { buf.as_mut_ptr() as *mut f32 };

    #[cfg(target_arch = "x86_64")]
    {
        // Fast path: non-temporal stores for aligned 32-byte regions.
        // Align the pointer upward to 32 bytes; fill the prefix with regular stores.
        let addr      = ptr as usize;
        let align_off = (32 - addr % 32) % 32 / std::mem::size_of::<f32>();
        let prefix    = align_off.min(len);
        for i in 0..prefix {
            unsafe { ptr.add(i).write(value); }
        }
        let aligned_ptr  = unsafe { ptr.add(prefix) };
        let aligned_len  = len - prefix;
        let nt_len       = aligned_len / 8 * 8; // round down to multiple of 8

        if nt_len > 0 {
            // SAFETY: aligned_ptr is 32-byte aligned, nt_len is multiple of 8.
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::alloc::fill_nontemporal_f32(aligned_ptr, nt_len, value);
                }
            } else {
                let slice = unsafe { std::slice::from_raw_parts_mut(aligned_ptr, nt_len) };
                slice.par_iter_mut().for_each(|x| *x = value);
            }
        }
        // Scalar tail
        for i in (prefix + nt_len)..len {
            unsafe { ptr.add(i).write(value); }
        }
        return Ok(());
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        slice.par_iter_mut().for_each(|x| *x = value);
        return Ok(());
    }
}

// ─── Parallel strided binary zip ─────────────────────────────────────────────

/// Applies `f(a[i], b[i]) -> T` over all elements of two same-shape C-contiguous
/// buffers, writing results into `dst`.  Generalises element-wise arithmetic.
pub fn parallel_zip<S, D, F>(
    a:   &Buffer,
    b:   &Buffer,
    dst: &mut Buffer,
    f:   F,
) -> MohuResult<()>
where
    S: Scalar + Copy + Send + Sync,
    D: Scalar + Copy + Send + Sync,
    F: Fn(S, S) -> D + Send + Sync,
{
    if S::DTYPE != a.dtype() || S::DTYPE != b.dtype() {
        return Err(MohuError::DTypeMismatch {
            expected: S::DTYPE.to_string(),
            got:      a.dtype().to_string(),
        });
    }
    if a.len() != b.len() || a.len() != dst.len() {
        return Err(MohuError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got:      b.shape().to_vec(),
        });
    }
    if !a.is_c_contiguous() || !b.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }
    if !dst.is_writeable() { return Err(MohuError::ReadOnly); }
    dst.make_unique()?;

    let len   = a.len();
    let a_s   = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const S, len) };
    let b_s   = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const S, len) };
    let d_ptr = unsafe { dst.as_mut_ptr() } as *mut D;
    let d_s   = unsafe { std::slice::from_raw_parts_mut(d_ptr, len) };

    a_s.par_iter()
        .zip(b_s.par_iter())
        .zip(d_s.par_iter_mut())
        .for_each(|((a, b), d)| *d = f(*a, *b));

    Ok(())
}

use num_traits;
