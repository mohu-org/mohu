#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
use crate::detect::cpu_features;

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn copy_f32_avx2(dst: *mut f32, src: *const f32, len: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_loadu_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: this chunk is within both caller-valid ranges; unaligned load/store have no alignment precondition.
        let v = unsafe { _mm256_loadu_ps(src.add(i)) };
        unsafe { _mm256_storeu_ps(dst.add(i), v) };
        i += 8;
    }
    // SAFETY: tail is in-bounds and source/destination are non-overlapping by contract.
    for j in i..len {
        // SAFETY: the public contract guarantees each source/destination
        // element lies in its respective valid range; unaligned access is
        // intentional and does not require typed-pointer alignment.
        let value = unsafe { src.add(j).read_unaligned() };
        unsafe { dst.add(j).write_unaligned(value) };
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn copy_f64_avx2(dst: *mut f64, src: *const f64, len: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_loadu_pd, _mm256_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};
    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: this chunk is within both caller-valid ranges; unaligned load/store have no alignment precondition.
        let v = unsafe { _mm256_loadu_pd(src.add(i)) };
        unsafe { _mm256_storeu_pd(dst.add(i), v) };
        i += 4;
    }
    // SAFETY: tail is in-bounds and source/destination are non-overlapping by contract.
    for j in i..len {
        // SAFETY: the public contract guarantees each source/destination
        // element lies in its respective valid range; unaligned access is
        // intentional and does not require typed-pointer alignment.
        let value = unsafe { src.add(j).read_unaligned() };
        unsafe { dst.add(j).write_unaligned(value) };
    }
}

/// Copy `len` `f32` values from `src` to `dst`.
///
/// # Safety
/// `src` must be valid for reads and `dst` for writes of `len` consecutive
/// values, and the ranges must not overlap. No element is accessed at zero length.
pub unsafe fn copy_f32(dst: *mut f32, src: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { copy_f32_avx2(dst, src, len) };
        return;
    }
    // SAFETY: caller contract exactly matches `copy_nonoverlapping`.
    for i in 0..len {
        // SAFETY: the public contract guarantees each element is in-bounds;
        // unaligned access is intentional.
        let value = unsafe { src.add(i).read_unaligned() };
        unsafe { dst.add(i).write_unaligned(value) };
    }
}

/// Copy `len` `f64` values from `src` to `dst`.
///
/// # Safety
/// `src` must designate memory for `len` reads and `dst` memory for `len`
/// writes. The ranges must not overlap. No element is accessed at zero length.
pub unsafe fn copy_f64(dst: *mut f64, src: *const f64, len: usize) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { copy_f64_avx2(dst, src, len) };
        return;
    }
    // SAFETY: caller contract exactly matches `copy_nonoverlapping`.
    for i in 0..len {
        // SAFETY: the public contract guarantees each element is in-bounds;
        // unaligned access is intentional.
        let value = unsafe { src.add(i).read_unaligned() };
        unsafe { dst.add(i).write_unaligned(value) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    #[test]
    fn copies_all_boundaries() {
        for &len in &[0, 1, 3, 4, 5, 7, 8, 9, 4097, 65537] {
            let src32: Vec<f32> = (0..len).map(|i| i as f32 + 0.25).collect();
            let src64: Vec<f64> = (0..len).map(|i| i as f64 - 7.5).collect();
            let mut dst32 = vec![0.0; len];
            let mut dst64 = vec![0.0; len];
            // SAFETY: vectors provide valid non-overlapping ranges; dangling
            // pointers are used only for zero-length calls.
            unsafe {
                copy_f32(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        dst32.as_mut_ptr()
                    },
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        src32.as_ptr()
                    },
                    len,
                );
                copy_f64(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        dst64.as_mut_ptr()
                    },
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        src64.as_ptr()
                    },
                    len,
                );
            }
            assert_eq!(src32, dst32);
            assert_eq!(src64, dst64);
        }
    }

    #[test]
    fn copies_valid_unaligned_ranges() {
        let len = 65;
        let src = (0..len).map(|i| i as f64 + 0.125).collect::<Vec<_>>();
        let mut src_bytes = vec![0u8; len * 8 + 1];
        let mut dst_bytes = vec![0u8; len * 8 + 1];
        // SAFETY: offset one leaves valid non-overlapping ranges of `len`
        // f64 values, and the implementation uses unaligned loads/stores.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr().cast::<u8>(),
                src_bytes.as_mut_ptr().add(1),
                len * 8,
            );
            copy_f64(
                dst_bytes.as_mut_ptr().add(1).cast(),
                src_bytes.as_ptr().add(1).cast(),
                len,
            );
            let result = (0..len)
                .map(|i| {
                    dst_bytes
                        .as_ptr()
                        .add(1)
                        .cast::<f64>()
                        .add(i)
                        .read_unaligned()
                })
                .collect::<Vec<_>>();
            assert_eq!(result, src);
        }
    }
}
