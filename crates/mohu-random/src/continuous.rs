// continuous — implementation pending

use mohu_buffer::Buffer;
use mohu_error::{MohuError, MohuResult};
use rand::Rng;

pub fn uniform(shape: &[usize], low: f64, high: f64) -> MohuResult<Buffer> {
    if high <= low {
        return Err(MohuError::domain(
            "uniform",
            "high must be greater than low",
        ));
    }

    let n: usize = shape.iter().product();

    let mut rng = rand::rng();

    let data: Vec<f64> = (0..n)
        .map(|_| rng.random_range(low..high))
        .collect();

    let buf = Buffer::from_vec(data)?;
    buf.reshape(shape)
}

pub fn normal(shape: &[usize], mean: f64, std: f64) -> MohuResult<Buffer> {
    if std <= 0.0 {
        return Err(MohuError::domain(
            "normal",
            "std must be positive",
        ));
    }

    let n: usize = shape.iter().product();
    let mut rng = rand::rng();

    let data: Vec<f64> = (0..n)
        .map(|_| {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();

            let z0 =
                (-2.0 * u1.ln()).sqrt()
                * (2.0 * std::f64::consts::PI * u2).cos();

            mean + std * z0
        })
        .collect();

    let buf = Buffer::from_vec(data)?;
    buf.reshape(shape)
}