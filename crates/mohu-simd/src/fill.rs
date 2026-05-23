// SAFETY: The safety invariants are enforced by the caller.

macro_rules! impl_fill {
    ($fn_name:ident, $fn_avx2:ident, $fn_scalar:ident, $ty:ty, $chunk_size:expr, $avx2_set:ident, $avx2_store:ident) => {
        /// Fills a buffer with a scalar value.
        ///
        /// # Safety
        /// * `dst` must be valid for writes of `len` elements.
        pub unsafe fn $fn_name(dst: *mut $ty, len: usize, val: $ty) {
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: The feature is checked, and caller preconditions apply.
                    unsafe {
                        return $fn_avx2(dst, len, val);
                    }
                }
            }
            // SAFETY: Caller preconditions apply.
            unsafe {
                $fn_scalar(dst, len, val);
            }
        }

        // SAFETY: Caller must ensure `dst` is valid for `len` elements.
        #[inline(always)]
        unsafe fn $fn_scalar(mut dst: *mut $ty, len: usize, val: $ty) {
            for _ in 0..len {
                // SAFETY: The bounds and validity of `dst` are guaranteed by the caller.
                dst.write(val);
                dst = dst.add(1);
            }
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
        #[target_feature(enable = "avx2")]
        // SAFETY: Caller must ensure `dst` is valid for `len` elements and AVX2 is available.
        unsafe fn $fn_avx2(mut dst: *mut $ty, mut len: usize, val: $ty) {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            
            // SAFETY: AVX2 feature is detected before calling this function.
            let vec_val = $avx2_set(val);
            
            while len >= $chunk_size {
                // SAFETY: The caller ensures that `dst` is valid for writes.
                $avx2_store(dst as *mut _, vec_val);
                dst = dst.add($chunk_size);
                len -= $chunk_size;
            }
            
            // Handle remaining elements with scalar fallback
            // SAFETY: Remaining bounds are valid.
            $fn_scalar(dst, len, val);
        }
    };
}

impl_fill!(fill_f32, fill_f32_avx2, fill_f32_scalar, f32, 8, _mm256_set1_ps, _mm256_storeu_ps);
impl_fill!(fill_f64, fill_f64_avx2, fill_f64_scalar, f64, 4, _mm256_set1_pd, _mm256_storeu_pd);
impl_fill!(fill_i32, fill_i32_avx2, fill_i32_scalar, i32, 8, _mm256_set1_epi32, _mm256_storeu_si256);
impl_fill!(fill_i64, fill_i64_avx2, fill_i64_scalar, i64, 4, _mm256_set1_epi64x, _mm256_storeu_si256);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_f32() {
        let mut buf = vec![0.0f32; 10];
        unsafe { fill_f32(buf.as_mut_ptr(), buf.len(), 42.0) };
        assert!(buf.iter().all(|&x| x == 42.0));
    }

    #[test]
    fn test_fill_f64_large() {
        let mut buf = vec![0.0f64; 5000];
        unsafe { fill_f64(buf.as_mut_ptr(), buf.len(), 3.14) };
        assert!(buf.iter().all(|&x| x == 3.14));
    }

    #[test]
    fn test_fill_i32_length_1() {
        let mut buf = vec![0i32; 1];
        unsafe { fill_i32(buf.as_mut_ptr(), buf.len(), -10) };
        assert_eq!(buf[0], -10);
    }

    #[test]
    fn test_fill_i64() {
        let mut buf = vec![0i64; 15];
        unsafe { fill_i64(buf.as_mut_ptr(), buf.len(), 100) };
        assert!(buf.iter().all(|&x| x == 100));
    }
}
