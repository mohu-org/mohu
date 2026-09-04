#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
use crate::detect::cpu_features;

unsafe fn fill_scalar<T: Copy>(dst: *mut T, len: usize, value: T) {
    for i in 0..len {
        // SAFETY: the public contract guarantees exclusive validity for this range.
        unsafe { dst.add(i).write_unaligned(value) };
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_f32_avx2(dst: *mut f32, len: usize, value: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_set1_ps, _mm256_storeu_ps};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_set1_ps, _mm256_storeu_ps};
    let v = _mm256_set1_ps(value);
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: this eight-element chunk is within the caller-valid range; storeu has no alignment precondition.
        unsafe { _mm256_storeu_ps(dst.add(i), v) };
        i += 8;
    }
    // SAFETY: the remaining tail is within the caller-valid range.
    unsafe { fill_scalar(dst.add(i), len - i, value) };
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_f64_avx2(dst: *mut f64, len: usize, value: f64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_set1_pd, _mm256_storeu_pd};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_set1_pd, _mm256_storeu_pd};
    let v = _mm256_set1_pd(value);
    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: this four-element chunk is within the caller-valid range; storeu has no alignment precondition.
        unsafe { _mm256_storeu_pd(dst.add(i), v) };
        i += 4;
    }
    // SAFETY: the remaining tail is within the caller-valid range.
    unsafe { fill_scalar(dst.add(i), len - i, value) };
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_i32_avx2(dst: *mut i32, len: usize, value: i32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_set1_epi32, _mm256_storeu_si256};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_set1_epi32, _mm256_storeu_si256};
    let v = _mm256_set1_epi32(value);
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: this eight-element chunk is within the caller-valid range; storeu has no alignment precondition.
        unsafe { _mm256_storeu_si256(dst.add(i).cast(), v) };
        i += 8;
    }
    // SAFETY: the remaining tail is within the caller-valid range.
    unsafe { fill_scalar(dst.add(i), len - i, value) };
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn fill_i64_avx2(dst: *mut i64, len: usize, value: i64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm256_set1_epi64x, _mm256_storeu_si256};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm256_set1_epi64x, _mm256_storeu_si256};
    let v = _mm256_set1_epi64x(value);
    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: this four-element chunk is within the caller-valid range; storeu has no alignment precondition.
        unsafe { _mm256_storeu_si256(dst.add(i).cast(), v) };
        i += 4;
    }
    // SAFETY: the remaining tail is within the caller-valid range.
    unsafe { fill_scalar(dst.add(i), len - i, value) };
}

/// Fill `len` consecutive `f32` values.
///
/// # Safety
/// For `len > 0`, `dst` must be valid for exclusive writes of `len` consecutive
/// values for the duration of this call. The range may be unaligned because
/// the implementation uses unaligned accesses. For `len == 0`, no element
/// is accessed.
pub unsafe fn fill_f32(dst: *mut f32, len: usize, value: f32) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { fill_f32_avx2(dst, len, value) };
        return;
    }
    // SAFETY: caller contract is exactly the scalar helper precondition.
    unsafe { fill_scalar(dst, len, value) };
}

/// Fill `len` consecutive `f64` values.
///
/// # Safety
/// For `len > 0`, `dst` must designate memory for exclusive writes of `len`
/// consecutive values for the duration of this call. The range may be unaligned
/// because the implementation uses unaligned accesses. For `len == 0`, no
/// element is accessed.
pub unsafe fn fill_f64(dst: *mut f64, len: usize, value: f64) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { fill_f64_avx2(dst, len, value) };
        return;
    }
    // SAFETY: caller contract is exactly the scalar helper precondition.
    unsafe { fill_scalar(dst, len, value) };
}

/// Fill `len` consecutive `i32` values.
///
/// # Safety
/// For `len > 0`, `dst` must designate memory for exclusive writes of `len`
/// consecutive values for the duration of this call. The range may be unaligned
/// because the implementation uses unaligned accesses. For `len == 0`, no
/// element is accessed.
pub unsafe fn fill_i32(dst: *mut i32, len: usize, value: i32) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { fill_i32_avx2(dst, len, value) };
        return;
    }
    // SAFETY: caller contract is exactly the scalar helper precondition.
    unsafe { fill_scalar(dst, len, value) };
}

/// Fill `len` consecutive `i64` values.
///
/// # Safety
/// For `len > 0`, `dst` must designate memory for exclusive writes of `len`
/// consecutive values for the duration of this call. The range may be unaligned
/// because the implementation uses unaligned accesses. For `len == 0`, no
/// element is accessed.
pub unsafe fn fill_i64(dst: *mut i64, len: usize, value: i64) {
    if len == 0 {
        return;
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
    if cpu_features().avx2 {
        // SAFETY: runtime detection proves AVX2; caller contract is forwarded.
        unsafe { fill_i64_avx2(dst, len, value) };
        return;
    }
    // SAFETY: caller contract is exactly the scalar helper precondition.
    unsafe { fill_scalar(dst, len, value) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::NonNull;

    #[test]
    fn fills_all_types_and_boundaries() {
        for &len in &[0, 1, 3, 4, 5, 7, 8, 9, 63, 64, 65, 4097, 65537] {
            let mut f32s = vec![0.0f32; len];
            let mut f64s = vec![0.0f64; len];
            let mut i32s = vec![0i32; len];
            let mut i64s = vec![0i64; len];
            // SAFETY: each vector is valid for `len` exclusive writes; the
            // dangling pointers are not accessed for the zero-length calls.
            unsafe {
                fill_f32(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        f32s.as_mut_ptr()
                    },
                    len,
                    1.25,
                );
                fill_f64(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        f64s.as_mut_ptr()
                    },
                    len,
                    -3.5,
                );
                fill_i32(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        i32s.as_mut_ptr()
                    },
                    len,
                    0x12345678,
                );
                fill_i64(
                    if len == 0 {
                        NonNull::dangling().as_ptr()
                    } else {
                        i64s.as_mut_ptr()
                    },
                    len,
                    -0x123456789,
                );
            }
            assert!(f32s.iter().all(|&x| x == 1.25));
            assert!(f64s.iter().all(|&x| x == -3.5));
            assert!(i32s.iter().all(|&x| x == 0x12345678));
            assert!(i64s.iter().all(|&x| x == -0x123456789));
        }
    }

    #[test]
    fn fills_valid_unaligned_ranges() {
        let len = 65;
        let mut bytes = vec![0u8; len * 4 + 1];
        // SAFETY: offset one leaves exactly `len * size_of::<f32>()` bytes;
        // the fill implementation intentionally uses unaligned stores.
        unsafe { fill_f32(bytes.as_mut_ptr().add(1).cast(), len, 2.5) };
        let values = (0..len)
            .map(|i| unsafe { bytes.as_ptr().add(1).cast::<f32>().add(i).read_unaligned() })
            .collect::<Vec<_>>();
        assert!(values.iter().all(|&x| x == 2.5));
    }
}
