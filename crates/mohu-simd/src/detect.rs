use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
}

impl CpuFeatures {
    #[allow(clippy::needless_return)]
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
                neon: false,
                sse4_1: std::is_x86_feature_detected!("sse4.1"),
                sse4_2: std::is_x86_feature_detected!("sse4.2"),
            };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self {
                avx2: false,
                avx512f: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                sse4_1: false,
                sse4_2: false,
            };
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
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

pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detection_is_stable() {
        assert_eq!(CpuFeatures::detect(), CpuFeatures::detect());
        assert!(std::ptr::eq(cpu_features(), cpu_features()));
    }
}
