use mohu_buffer::{buffer::Buffer, layout::Order};
use mohu_dtype::{
    dispatch_numeric,
    dtype::DType,
    promote::{CastMode, promote},
    scalar::Scalar,
};
use mohu_error::{MohuError, MohuResult};

pub fn matmul(lhs: &Buffer, rhs: &Buffer) -> MohuResult<Buffer> {
    let lhs_shape = lhs.shape().to_vec();
    let rhs_shape = rhs.shape().to_vec();

    // Normalize every supported case into an effective (m, k, n) plus the
    // real output shape the caller should see.
    let (m, k, n, out_shape): (usize, usize, usize, Vec<usize>) =
        match (lhs_shape.len(), rhs_shape.len()) {
            // (M, K) x (K, N) -> (M, N)
            (2, 2) => {
                let (m, k) = (lhs_shape[0], lhs_shape[1]);
                let (k2, n) = (rhs_shape[0], rhs_shape[1]);
                check_k(k, k2)?;
                (m, k, n, vec![m, n])
            },
            // (K,) as (1, K); x (K, N) -> (N,)
            (1, 2) => {
                let k = lhs_shape[0];
                let (k2, n) = (rhs_shape[0], rhs_shape[1]);
                check_k(k, k2)?;
                (1, k, n, vec![n])
            },
            // (M, K) x (K,) as (K, 1) -> (M,)
            (2, 1) => {
                let (m, k) = (lhs_shape[0], lhs_shape[1]);
                let k2 = rhs_shape[0];
                check_k(k, k2)?;
                (m, k, 1, vec![m])
            },
            // (K,) x (K,) -> scalar, shape []
            (1, 1) => {
                let k = lhs_shape[0];
                let k2 = rhs_shape[0];
                check_k(k, k2)?;
                (1, k, 1, vec![])
            },
            (a, b) => {
                // TODO: confirm the actual MohuError variant name for this —
                // couldn't find `pub enum MohuError` in lib.rs via grep, so it's
                // likely declared elsewhere (error.rs?) or via a macro.
                return Err(MohuError::DimensionMismatch {
                    expected: 2,
                    got: a.max(b),
                });
            },
        };

    // Spec: promote dtypes, but integer results must promote to F64.
    let promoted = promote(lhs.dtype(), rhs.dtype());
    let out_dtype = if promoted.is_integer() {
        DType::F64
    } else {
        promoted
    };

    // Cast both operands to out_dtype so the kernel only ever deals with one
    // type. This also guarantees a contiguous, row-major layout for the
    // index math below, since `cast` allocates fresh Order::C storage
    // (and short-circuits to `to_contiguous()` when no dtype change is needed).
    // TODO: confirm the right CastMode variant here (Safe/Checked/Lossy — whatever
    // the crate calls "just do a numeric cast", as opposed to a bit-reinterpret).
    let lhs_cast = lhs.cast(out_dtype, CastMode::Safe)?;
    let rhs_cast = rhs.cast(out_dtype, CastMode::Safe)?;

    let mut out = Buffer::alloc(out_dtype, &out_shape, Order::C)?;

    macro_rules! do_matmul {
        ($T:ty) => {
            matmul_typed::<$T>(&lhs_cast, &rhs_cast, &mut out, m, k, n)
        };
    }
    dispatch_numeric!(out_dtype, do_matmul)??;

    Ok(out)
}

#[inline]
fn check_k(k: usize, k2: usize) -> MohuResult<()> {
    if k != k2 {
        return Err(MohuError::ShapeMismatch {
            expected: vec![k],
            got: vec![k2],
        });
    }
    Ok(())
}

/// Naive O(M*K*N) matmul kernel over logical (m, k) x (k, n) shapes.
/// Assumes `lhs`, `rhs`, and `out` are contiguous and share dtype `T`
/// (guaranteed by the `cast` calls in `matmul`).
fn matmul_typed<T>(
    lhs: &Buffer,
    rhs: &Buffer,
    out: &mut Buffer,
    m: usize,
    k: usize,
    n: usize,
) -> MohuResult<()>
where
    T: Scalar + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let lhs_data = lhs.as_slice::<T>()?;
    let rhs_data = rhs.as_slice::<T>()?;
    let out_data = out.as_mut_slice::<T>()?;

    for i in 0..m {
        for j in 0..n {
            let mut sum = T::ZERO;
            // TODO: replace with BLAS dgemm/sgemm once the linear algebra
            // backend is available.
            for p in 0..k {
                sum = sum + lhs_data[i * k + p] * rhs_data[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    Ok(())
}
