// SAFETY: The safety invariants are enforced by the caller.

macro_rules! impl_copy {
    ($fn_name:ident, $fn_avx2:ident, $fn_scalar:ident, $ty:ty, $chunk_size:expr, $avx2_load:ident, $avx2_store:ident) => {
        /// Copies a buffer.
        ///
        /// # Safety
        /// * `dst` must be valid for writes of `len` elements.
        /// * `src` must be valid for reads of `len` elements.
        /// * `dst` and `src` must not overlap.
        pub unsafe fn $fn_name(dst: *mut $ty, src: *const $ty, len: usize) {
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: Feature checked and caller ensures validity of pointers.
                    unsafe {
                        return $fn_avx2(dst, src, len);
                    }
                }
            }
            // SAFETY: Caller preconditions apply.
            unsafe {
                $fn_scalar(dst, src, len);
            }
        }

        // SAFETY: Caller must ensure `dst` and `src` are valid for `len` elements.
        #[inline(always)]
        unsafe fn $fn_scalar(dst: *mut $ty, src: *const $ty, len: usize) {
            // SAFETY: Caller guarantees valid, non-overlapping pointers.
            core::ptr::copy_nonoverlapping(src, dst, len);
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx2"))]
        #[target_feature(enable = "avx2")]
        // SAFETY: Caller must ensure valid pointers and AVX2 availability.
        unsafe fn $fn_avx2(mut dst: *mut $ty, mut src: *const $ty, mut len: usize) {
            #[cfg(target_arch = "x86")]
            use core::arch::x86::*;
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::*;
            
            while len >= $chunk_size {
                // SAFETY: The caller ensures pointers are valid.
                let vec_val = $avx2_load(src as *const _);
                $avx2_store(dst as *mut _, vec_val);
                dst = dst.add($chunk_size);
                src = src.add($chunk_size);
                len -= $chunk_size;
            }
            
            // Handle remaining elements with scalar fallback
            // SAFETY: Remaining bounds are valid.
            $fn_scalar(dst, src, len);
        }
    };
}

impl_copy!(copy_f32, copy_f32_avx2, copy_f32_scalar, f32, 8, _mm256_loadu_ps, _mm256_storeu_ps);
impl_copy!(copy_f64, copy_f64_avx2, copy_f64_scalar, f64, 4, _mm256_loadu_pd, _mm256_storeu_pd);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_f32() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut dst = vec![0.0f32; 10];
        unsafe { copy_f32(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(src, dst);
    }

    #[test]
    fn test_copy_f64_large() {
        let src = vec![2.71f64; 5000];
        let mut dst = vec![0.0f64; 5000];
        unsafe { copy_f64(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(src, dst);
    }
    
    #[test]
    fn test_copy_f32_length_1() {
        let src = vec![42.0f32];
        let mut dst = vec![0.0f32];
        unsafe { copy_f32(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
        assert_eq!(src, dst);
    }
}
