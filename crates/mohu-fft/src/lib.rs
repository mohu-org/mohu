/// Fast Fourier Transforms for mohu.
///
/// Wraps `rustfft` for complex transforms over Buffer values.
///
/// Implemented here: C64/C128 FFT and inverse FFT along one arbitrary axis,
/// including normalization, zero-padding, and truncation. Real-input transforms,
/// frequency helpers, multidimensional transforms, and parallel batch execution
/// remain tracked separately.
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
pub use transform::{fft, ifft};
