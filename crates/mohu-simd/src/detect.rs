//! Runtime CPU feature detection for SIMD kernel dispatch.

use std::sync::OnceLock;

/// SIMD instruction sets available on the current host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
}

impl CpuFeatures {
    /// Detects available features at runtime for the current target architecture.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                neon: false,
                sse4_1: std::arch::is_x86_feature_detected!("sse4.1"),
                sse4_2: std::arch::is_x86_feature_detected!("sse4.2"),
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                sse4_1: false,
                sse4_2: false,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                avx512f: false,
                neon: false,
                sse4_1: false,
                sse4_2: false,
            }
        }
    }
}

static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Lazily-initialised global feature set. Safe to call from any thread after first use.
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

#[cfg(test)]
mod tests {
    use super::{cpu_features, CpuFeatures};

    #[test]
    fn detect_returns_struct_without_panicking() {
        let features = CpuFeatures::detect();
        let _ = (features.avx2, features.avx512f, features.neon);
    }

    #[test]
    fn global_cpu_features_is_initialized_once() {
        let a = cpu_features();
        let b = cpu_features();
        assert!(std::ptr::eq(a, b));
    }
}
