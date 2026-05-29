/// Fast Fourier Transforms for mohu.
///
/// Wraps `rustfft` for the core Cooley-Tukey algorithm and adds:
/// - N-dimensional FFT / IFFT over arbitrary axes
/// - Real-input FFT (`rfft`) producing Hermitian-symmetric output
/// - Frequency axis helpers (`fftfreq`, `rfftfreq`, `fftshift`)
/// - Parallel batch transforms via Rayon
///
/// # Equivalents
///
/// | mohu-fft function     | numpy.fft equivalent  |
/// |-----------------------|-----------------------|
/// | `fft(a, n, axis)`     | `np.fft.fft`          |
/// | `ifft(a, n, axis)`    | `np.fft.ifft`         |
/// | `rfft(a, n, axis)`    | `np.fft.rfft`         |
/// | `irfft(a, n, axis)`   | `np.fft.irfft`        |
/// | `fft2(a, s, axes)`    | `np.fft.fft2`         |
/// | `fftn(a, s, axes)`    | `np.fft.fftn`         |
/// | `fftfreq(n, d)`       | `np.fft.fftfreq`      |
/// | `fftshift(a, axes)`   | `np.fft.fftshift`     |
///
/// # Normalization modes
///
/// | Mode       | Forward scale  | Backward scale   |
/// |------------|----------------|------------------|
/// | `Backward` | 1              | 1/n (default)    |
/// | `Ortho`    | 1/sqrt(n)      | 1/sqrt(n)        |
/// | `Forward`  | 1/n            | 1                |
pub mod freq;
pub mod helpers;
pub mod nd;
pub mod norm;
pub mod plan;
pub mod real;
pub mod transform;

pub use mohu_error::{MohuError, MohuResult};
pub use norm::Norm;
