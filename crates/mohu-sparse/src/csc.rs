/// Compressed Sparse Column (CSC) sparse matrix format.
///
/// Stores non-zero elements compressed by column.
/// Best for column slicing and sparse matrix-matrix multiplication.

use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Compressed Sparse Column matrix.
pub struct CscMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    values: Vec<T>,
    row_indices: Vec<usize>,
    col_ptr: Vec<usize>,
}

impl<T: Scalar> CscMatrix<T> {
    /// Create a new CSC matrix from CSC components.
    pub fn new(values: Vec<T>, row_indices: Vec<usize>, col_ptr: Vec<usize>, rows: usize, cols: usize) -> MohuResult<Self> {
        if col_ptr.len() != cols + 1 {
            return Err(mohu_error::MohuError::ShapeMismatch {
                expected: vec![cols + 1],
                got: vec![col_ptr.len()],
            });
        }
        
        if values.len() != row_indices.len() {
            return Err(mohu_error::MohuError::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![row_indices.len()],
            });
        }

        Ok(Self {
            rows,
            cols,
            values,
            row_indices,
            col_ptr,
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

    /// Get the row indices.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get the column pointers.
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_creation() {
        let values = vec![1.0, 2.0, 3.0];
        let row_indices = vec![0, 1, 2];
        let col_ptr = vec![0, 2, 3];
        
        let csc = CscMatrix::new(values, row_indices, col_ptr, 3, 2).unwrap();
        
        assert_eq!(csc.shape(), (3, 2));
        assert_eq!(csc.nnz(), 3);
    }
}

