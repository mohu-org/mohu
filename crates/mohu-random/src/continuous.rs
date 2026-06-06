use mohu_buffer::Buffer;
use mohu_error::{MohuError, MohuResult, ensure};
use rand::Rng;

pub fn uniform(shape: &[usize], low: f64, high: f64) -> MohuResult<Buffer> {
    ensure!(
        low.is_finite() && high.is_finite() && high > low,
        MohuError::domain(
            "uniform",
            "high must be greater than low and bounds must be finite"
        )
    );

    let n: usize = shape.iter().product();
    let mut rng = rand::rng();

    let data: Vec<f64> = (0..n).map(|_| rng.random_range(low..high)).collect();

    Buffer::from_vec(data)?.reshape(shape)
}

pub fn normal(shape: &[usize], mean: f64, std: f64) -> MohuResult<Buffer> {
    ensure!(
        std.is_finite() && std > 0.0,
        MohuError::domain("normal", "std must be positive and finite")
    );

    let n: usize = shape.iter().product();
    let mut rng = rand::rng();

    let data: Vec<f64> = (0..n)
        .map(|_| {
            // Draw u1 from (0, 1] to avoid log(0) = -inf in Box-Muller
            let u1: f64 = loop {
                let x: f64 = rng.random();
                if x > 0.0 {
                    break x;
                }
            };
            let u2: f64 = rng.random();

            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            mean + std * z0
        })
        .collect();

    Buffer::from_vec(data)?.reshape(shape)
}
