/// Compressed Sparse Row (CSR) sparse matrix format.
///
/// Stores non-zero elements compressed by row.
/// Best for row slicing and sparse matrix-vector multiplication.

use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Compressed Sparse Row matrix.
pub struct CsrMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Create a new CSR matrix from CSR components.
    pub fn new(values: Vec<T>, col_indices: Vec<usize>, row_ptr: Vec<usize>, rows: usize, cols: usize) -> MohuResult<Self> {
        if row_ptr.len() != rows + 1 {
            return Err(mohu_error::MohuError::ShapeMismatch {
                expected: vec![rows + 1],
                got: vec![row_ptr.len()],
            });
        }
        
        if values.len() != col_indices.len() {
            return Err(mohu_error::MohuError::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![col_indices.len()],
            });
        }

        Ok(Self {
            rows,
            cols,
            values,
            col_indices,
            row_ptr,
        })
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get the shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the values.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Get the column indices.
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get the row pointers.
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptr = vec![0, 2, 3];
        
        let csr = CsrMatrix::new(values, col_indices, row_ptr, 2, 3).unwrap();
        
        assert_eq!(csr.shape(), (2, 3));
        assert_eq!(csr.nnz(), 3);
    }
}

