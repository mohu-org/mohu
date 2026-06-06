use mohu_buffer::Buffer;
use mohu_error::{MohuError, MohuResult, ensure};
use rand::Rng;

pub fn integers(shape: &[usize], low: i64, high: i64) -> MohuResult<Buffer> {
    ensure!(
        high > low,
        MohuError::domain("integers", "high must be greater than low")
    );

    let n: usize = shape.iter().product();
    let mut rng = rand::rng();

    let data: Vec<i64> = (0..n).map(|_| rng.random_range(low..high)).collect();

    Buffer::from_vec(data)?.reshape(shape)
}
