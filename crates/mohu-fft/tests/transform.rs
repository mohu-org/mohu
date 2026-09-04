use mohu_buffer::Buffer;
use mohu_dtype::DType;
use mohu_error::MohuError;
use mohu_fft::{Norm, fft, ifft};
use num_complex::Complex;
fn c64(v: &[Complex<f32>], s: &[usize]) -> Buffer {
    Buffer::from_slice(v).unwrap().reshape(s).unwrap()
}
fn c128(v: &[Complex<f64>], s: &[usize]) -> Buffer {
    Buffer::from_slice(v).unwrap().reshape(s).unwrap()
}
fn close(a: Complex<f64>, b: Complex<f64>) {
    assert!((a.re - b.re).abs() < 1e-9);
    assert!((a.im - b.im).abs() < 1e-9);
}
#[test]
fn c64_c128_roundtrip_norms() {
    let x = c64(&[Complex::new(1., 2.), Complex::new(3., -1.)], [2].as_ref());
    for n in [Norm::Backward, Norm::Ortho, Norm::Forward] {
        let y = ifft(&fft(&x, None, 0, n).unwrap(), None, 0, n).unwrap();
        for (a, b) in y
            .to_vec::<Complex<f32>>()
            .unwrap()
            .iter()
            .zip([Complex::new(1., 2.), Complex::new(3., -1.)])
        {
            assert!((a.re - b.re).abs() < 1e-5);
            assert!((a.im - b.im).abs() < 1e-5);
        }
    }
    let x = c128(&[Complex::new(1., 2.), Complex::new(3., -1.)], [2].as_ref());
    let y = fft(&x, None, 0, Norm::Backward).unwrap();
    assert_eq!(y.dtype(), DType::C128);
    close(y.to_vec::<Complex<f64>>().unwrap()[0], Complex::new(4., 1.));
}
#[test]
fn padding_truncation_and_axes() {
    let x = c128(
        &[
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
        ],
        [3].as_ref(),
    );
    assert_eq!(fft(&x, Some(5), 0, Norm::Backward).unwrap().shape(), [5]);
    assert_eq!(fft(&x, Some(2), 0, Norm::Backward).unwrap().shape(), [2]);
    let m = c128(
        &[
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
            Complex::new(4., 0.),
        ],
        [2, 2].as_ref(),
    );
    assert_eq!(fft(&m, None, 0, Norm::Backward).unwrap().shape(), [2, 2]);
}
#[test]
fn strided_and_empty_batch() {
    let x = c128(
        &[
            Complex::new(1., 0.),
            Complex::new(2., 0.),
            Complex::new(3., 0.),
            Complex::new(4., 0.),
        ],
        [2, 2].as_ref(),
    )
    .transpose();
    assert_eq!(fft(&x, None, 1, Norm::Backward).unwrap().shape(), [2, 2]);
    let e = c128(&[], [0, 4].as_ref());
    assert_eq!(fft(&e, None, 1, Norm::Backward).unwrap().shape(), [0, 4]);
    assert_eq!(fft(&e, Some(4), 0, Norm::Backward).unwrap().shape(), [4, 4]);
}
#[test]
fn invalid_inputs_error() {
    let x = c128(&[Complex::new(1., 0.)], [1].as_ref());
    assert!(matches!(
        fft(&x, None, 1, Norm::Backward),
        Err(MohuError::AxisOutOfRange { .. })
    ));
    assert!(matches!(
        fft(&x, Some(0), 0, Norm::Backward),
        Err(MohuError::ZeroSizedDimension { axis: 0 })
    ));
    let r = Buffer::zeros(DType::F64, [1].as_ref()).unwrap();
    assert!(matches!(
        fft(&r, None, 0, Norm::Backward),
        Err(MohuError::UnsupportedDType { .. })
    ));
    let s = c128(&[Complex::new(1., 0.)], &[]);
    assert!(matches!(
        fft(&s, None, 0, Norm::Backward),
        Err(MohuError::AxisOutOfRange { .. })
    ));
}
