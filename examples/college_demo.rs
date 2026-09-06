use mohu_array::NdArray;
use mohu_buffer::Buffer;
use mohu_ops::matmul::matmul;
use mohu_random::Rng;

fn main() -> mohu_buffer::MohuResult<()> {
    println!("============================================================");
    println!("MOHU — RUST-POWERED NUMERICAL ARRAYS");
    println!("============================================================");

    println!("\n1. ARRAY CONSTRUCTION");
    let array = NdArray::<f64>::from_shape_slice(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!("values  = {:?}", array.to_vec()?);
    println!("shape   = {:?}", array.shape());
    println!("ndim    = {}", array.ndim());
    println!("dtype   = {}", array.dtype());

    println!("\n2. VIEWS & STRIDES");
    let source = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?.reshape(&[2, 3])?;
    let reshaped = source.reshape(&[3, 2])?;
    let transposed = source.transpose();
    println!(
        "reshape  shape={:?} values={:?}",
        reshaped.shape(),
        reshaped.to_vec::<f64>()?
    );
    println!(
        "transpose shape={:?} strides={:?}",
        transposed.shape(),
        transposed.strides()
    );
    println!("transpose values={:?}", transposed.to_vec::<f64>()?);
    println!("zero-copy backing storage = {}", transposed.is_shared());

    println!("\n3. BROADCASTING");
    let row = Buffer::from_slice(&[10.0_f64, 20.0, 30.0])?;
    let broadcast = row.broadcast_to(&[2, 3])?;
    println!("shape   = {:?}", broadcast.shape());
    println!(
        "strides = {:?} (0 means repeated view)",
        broadcast.strides()
    );
    println!("values  = {:?}", broadcast.to_vec::<f64>()?);

    println!("\n4. MATRIX MULTIPLICATION");
    let lhs = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0])?.reshape(&[2, 2])?;
    let rhs = Buffer::from_slice(&[5.0_f64, 6.0, 7.0, 8.0])?.reshape(&[2, 2])?;
    let product = matmul(&lhs, &rhs)?;
    println!("[[1, 2], [3, 4]] @ [[5, 6], [7, 8]]");
    println!("result  = {:?}", product.to_vec::<f64>()?);
    assert_eq!(product.to_vec::<f64>()?, vec![19.0, 22.0, 43.0, 50.0]);

    println!("\n5. SEEDED RANDOMNESS");
    let mut first = Rng::new(42);
    let mut second = Rng::new(42);
    let first_values = first.integers(&[4], 0, 10)?;
    let second_values = second.integers(&[4], 0, 10)?;
    assert_eq!(
        first_values.to_vec::<i64>()?,
        second_values.to_vec::<i64>()?
    );
    println!("seed 42 values = {:?}", first_values.to_vec::<i64>()?);
    println!("same seed reproducible = true");

    println!("\n============================================================");
    println!("DEMO COMPLETE — tested Buffer, views, broadcasting, matmul, RNG");
    println!("============================================================");
    Ok(())
}
