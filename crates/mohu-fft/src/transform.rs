use crate::norm::Norm;
use mohu_buffer::{Buffer, NdIndexIter, Order};
use mohu_dtype::DType;
use mohu_error::{MohuError, MohuResult};
use num_complex::Complex;
use num_traits::{FromPrimitive, Zero};
use rustfft::FftPlanner;

fn validate(input: &Buffer, n: Option<usize>, axis: usize) -> MohuResult<usize> {
    if axis >= input.ndim() {
        return Err(MohuError::AxisOutOfRange {
            axis: axis as i64,
            ndim: input.ndim(),
            valid: if input.ndim() == 0 {
                "none".into()
            } else {
                format!("0..{}", input.ndim() - 1)
            },
        });
    }
    let len = n.unwrap_or(input.shape()[axis]);
    if len == 0 {
        return Err(MohuError::ZeroSizedDimension { axis });
    }
    if !matches!(input.dtype(), DType::C64 | DType::C128) {
        return Err(MohuError::UnsupportedDType {
            op: "fft",
            dtype: input.dtype().to_string(),
        });
    }
    Ok(len)
}
fn factor(norm: Norm, inverse: bool, n: usize) -> f64 {
    match (norm, inverse) {
        (Norm::Backward, false) | (Norm::Forward, true) => 1.0,
        (Norm::Backward, true) | (Norm::Forward, false) => 1.0 / n as f64,
        (Norm::Ortho, _) => 1.0 / (n as f64).sqrt(),
    }
}
macro_rules! transform_lines {
    ($name:ident, $real:ty) => {
        fn $name(
            input: &Buffer,
            output: &mut Buffer,
            n: usize,
            axis: usize,
            norm: Norm,
            inverse: bool,
        ) -> MohuResult<()> {
            let outer: Vec<usize> = input
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, &size)| (i != axis).then_some(size))
                .collect();
            let scale =
                <$real as FromPrimitive>::from_f64(factor(norm, inverse, n)).ok_or_else(|| {
                    MohuError::UnsupportedDType {
                        op: "fft normalization",
                        dtype: input.dtype().to_string(),
                    }
                })?;
            let mut planner = FftPlanner::<$real>::new();
            let plan = if inverse {
                planner.plan_fft_inverse(n)
            } else {
                planner.plan_fft_forward(n)
            };
            for coords in NdIndexIter::new(&outer) {
                let mut line = vec![Complex::<$real>::new(<$real>::zero(), <$real>::zero()); n];
                for (k, value) in line.iter_mut().enumerate().take(input.shape()[axis].min(n)) {
                    let mut index = coords.clone();
                    index.insert(axis, k);
                    *value = input.get(&index)?;
                }
                plan.process(&mut line);
                for value in &mut line {
                    *value *= scale;
                }
                for (k, value) in line.into_iter().enumerate() {
                    let mut index = coords.clone();
                    index.insert(axis, k);
                    output.set(&index, value)?;
                }
            }
            Ok(())
        }
    };
}
transform_lines!(transform_c64, f32);
transform_lines!(transform_c128, f64);

fn transform(
    input: &Buffer,
    n: Option<usize>,
    axis: usize,
    norm: Norm,
    inverse: bool,
) -> MohuResult<Buffer> {
    let n = validate(input, n, axis)?;
    let mut shape = input.shape().to_vec();
    shape[axis] = n;
    let mut out = Buffer::alloc(input.dtype(), &shape, Order::C)?;
    match input.dtype() {
        DType::C64 => transform_c64(input, &mut out, n, axis, norm, inverse)?,
        DType::C128 => transform_c128(input, &mut out, n, axis, norm, inverse)?,
        _ => unreachable!(),
    };
    Ok(out)
}
/// Computes a complex FFT independently along one Buffer axis.
pub fn fft(input: &Buffer, n: Option<usize>, axis: usize, norm: Norm) -> MohuResult<Buffer> {
    transform(input, n, axis, norm, false)
}
/// Computes the inverse complex FFT independently along one Buffer axis.
pub fn ifft(input: &Buffer, n: Option<usize>, axis: usize, norm: Norm) -> MohuResult<Buffer> {
    transform(input, n, axis, norm, true)
}
