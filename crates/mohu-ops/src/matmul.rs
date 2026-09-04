//! Buffer-facing matrix multiplication semantics.

use mohu_buffer::{Buffer, Order};
use mohu_dtype::{dispatch_numeric, dtype::DType, promote::CastMode, scalar::Scalar};
use mohu_error::{MohuError, MohuResult};

/// Multiplies one- or two-dimensional buffers without batch broadcasting.
///
/// Integer inputs produce `F64` output. Other numeric inputs use the current
/// promoted dtype. Operands may be strided; they are materialized as needed.
/// The private reference kernel is deliberately kept behind this semantic API
/// so a future `mohu-linalg` backend can replace it without changing callers.
pub fn matmul(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    let lhs_rank = lhs.ndim();
    let rhs_rank = rhs.ndim();
    let (m, k, n, output_shape) = match (lhs_rank, rhs_rank) {
        (2, 2) => {
            let (m, k) = (lhs.shape()[0], lhs.shape()[1]);
            let (rhs_k, n) = (rhs.shape()[0], rhs.shape()[1]);
            validate_k(k, rhs_k)?;
            (m, k, n, vec![m, n])
        },
        (1, 2) => {
            let k = lhs.shape()[0];
            let (rhs_k, n) = (rhs.shape()[0], rhs.shape()[1]);
            validate_k(k, rhs_k)?;
            (1, k, n, vec![n])
        },
        (2, 1) => {
            let (m, k) = (lhs.shape()[0], lhs.shape()[1]);
            let rhs_k = rhs.shape()[0];
            validate_k(k, rhs_k)?;
            (m, k, 1, vec![m])
        },
        (1, 1) => {
            let k = lhs.shape()[0];
            let rhs_k = rhs.shape()[0];
            validate_k(k, rhs_k)?;
            (1, k, 1, vec![])
        },
        (lhs_rank, rhs_rank) => {
            return Err(MohuError::DimensionMismatch {
                expected: 1,
                got: lhs_rank.max(rhs_rank),
            });
        },
    };

    let promoted = mohu_dtype::promote::promote(lhs.dtype(), rhs.dtype());
    let output_dtype = if promoted.is_integer() {
        DType::F64
    } else {
        promoted
    };
    let lhs = lhs.cast(output_dtype, CastMode::Safe)?;
    let rhs = rhs.cast(output_dtype, CastMode::Safe)?;
    let mut output = Buffer::alloc(output_dtype, &output_shape, Order::C)?;

    macro_rules! multiply {
        ($T:ty) => {{
            matmul_typed::<$T>(&lhs, &rhs, &mut output, m, k, n)?;
            Ok::<Buffer, MohuError>(output)
        }};
    }
    dispatch_numeric!(output_dtype, multiply)?
}

fn validate_k(lhs_k: usize, rhs_k: usize) -> MohuResult<()> {
    if lhs_k != rhs_k {
        return Err(MohuError::ShapeMismatch {
            expected: vec![lhs_k],
            got: vec![rhs_k],
        });
    }
    Ok(())
}

/// Reference O(M*K*N) kernel. Future optimized backends stay behind `matmul`.
fn matmul_typed<T>(
    lhs: &Buffer,
    rhs: &Buffer,
    output: &mut Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> MohuResult<()>
where
    T: Scalar + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let lhs_values = lhs.as_slice::<T>()?;
    let rhs_values = rhs.as_slice::<T>()?;
    let output_values = output.as_mut_slice::<T>()?;
    for row in 0..m {
        for column in 0..n {
            let mut sum = T::ZERO;
            for inner in 0..k {
                sum = sum + lhs_values[row * k + inner] * rhs_values[inner * n + column];
            }
            output_values[row * n + column] = sum;
        }
    }
    Ok(())
}
