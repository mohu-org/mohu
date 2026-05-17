//! Annotated walkthrough of the core [`Buffer`] API.
//!
//! Run from the workspace root:
//!
//! ```bash
//! cargo run -p mohu-core --example buffer_basics
//! ```

use mohu_buffer::{Buffer, SliceArg};
use mohu_dtype::DType;
use mohu_error::MohuResult;

fn main() -> MohuResult<()> {
    // ── 1. Creating buffers ───────────────────────────────────────────────────
    // NumPy: np.zeros((2, 3), dtype=float64)
    let mut grid = Buffer::zeros(DType::F64, &[2, 3])?;

    // NumPy: np.ones(4, dtype=float32)
    let ones = Buffer::ones(DType::F32, &[4])?;

    // NumPy: np.array([1, 2, 3, 4, 5, 6], dtype=float64).reshape(2, 3)
    let data = Buffer::from_slice(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let matrix = data.reshape(&[2, 3])?;

    // ── 2. Shape and dtype inspection ───────────────────────────────────────
    println!("matrix: shape={:?}, ndim={}, dtype={}", matrix.shape(), matrix.ndim(), matrix.dtype());
    println!("ones:   shape={:?}, dtype={}", ones.shape(), ones.dtype());

    // ── 3. Element access ───────────────────────────────────────────────────
    let corner: f64 = matrix.get(&[1, 2])?;
    println!("matrix[1, 2] = {corner}"); // 6.0

    grid.set(&[0, 0], 42.0)?;
    let updated: f64 = grid.get(&[0, 0])?;
    println!("grid[0, 0] after set = {updated}");

    // ── 4. Layout operations ────────────────────────────────────────────────
    // NumPy: matrix.T
    let transposed = matrix.transpose();
    println!("transpose shape={:?}", transposed.shape());

    // NumPy: matrix.reshape(3, 2)
    let reshaped = matrix.reshape(&[3, 2])?;
    println!("reshape(3, 2) shape={:?}", reshaped.shape());

    // NumPy: matrix[::2, :] — slice rows with step 2
    let every_other_row = matrix.slice_axis(
        0,
        SliceArg {
            start: Some(0),
            stop: None,
            step: Some(2),
        },
    )?;
    println!("slice_axis rows [::2] shape={:?}", every_other_row.shape());

    // ── 5. Reductions ─────────────────────────────────────────────────────────
    println!("sum_all_f64 = {}", matrix.sum_all_f64()?);
    println!("min_all_f64 = {}", matrix.min_all_f64()?);
    println!("max_all_f64 = {}", matrix.max_all_f64()?);

    Ok(())
}
