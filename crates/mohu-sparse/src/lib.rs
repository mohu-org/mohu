/// Sparse matrix formats and operations for mohu.
///
/// Provides the three canonical sparse storage formats — COO, CSR, CSC —
/// along with conversions between them, basic arithmetic, and a sparse
/// matrix-vector product (SpMV) kernel.
///
/// # Formats
///
/// | Module  | Format                        | Best for                    |
/// |---------|-------------------------------|-----------------------------|
/// | [`coo`] | Coordinate (i, j, value)      | Incremental construction    |
/// | [`csr`] | Compressed Sparse Row         | Row slicing, SpMV           |
/// | [`csc`] | Compressed Sparse Column      | Column slicing, SpMM        |
/// | [`bsr`] | Block Sparse Row              | Dense block sub-matrices    |
/// | [`dia`] | Diagonal                      | Banded / tridiagonal systems|
///
/// # Operations
///
/// | Module       | Operations                                          |
/// |--------------|-----------------------------------------------------|
/// | [`arith`]    | add, sub, mul (element-wise), scalar multiply        |
/// | [`spmv`]     | sparse matrix × dense vector                        |
/// | [`spmm`]     | sparse matrix × dense matrix                        |
/// | [`convert`]  | conversions between all format pairs                 |
/// | [`mod@slice`]    | row / column slicing                                 |
/// | [`linalg`]   | triangular solve, norm, condest                     |
///
/// # Construction
///
/// ```rust,ignore
/// let mut coo = CooMatrix::<f64>::new(1000, 1000);
/// coo.push(0, 1, 3.14);
/// coo.push(5, 7, 2.71);
/// let csr = CsrMatrix::from(coo);
/// ```
pub mod arith;
pub mod bsr;
pub mod convert;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod dia;
pub mod linalg;
pub mod slice;
pub mod spmm;
pub mod spmv;

pub use mohu_error::{MohuError, MohuResult};
