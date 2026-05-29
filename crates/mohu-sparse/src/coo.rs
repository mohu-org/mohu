/// Coordinate (COO) sparse matrix format.
///
/// Stores non-zero elements as (row, col, value) tuples.
/// Best for incremental construction and format conversion.

use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Coordinate sparse matrix.
pub struct CooMatrix<T: Scalar> {
    rows: usize,
    cols: usize,
    data: Vec<(usize, usize, T)>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Create a new empty COO matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: Vec::new(),
        }
    }

    /// Add a non-zero element.
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            self.data.push((row, col, value));
        }
    }

    /// Get the number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get the shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get the data as a slice of (row, col, value) tuples.
    pub fn data(&self) -> &[(usize, usize, T)] {
        &self.data
    }

    /// Sort the data by row, then by column.
    pub fn sort(&mut self) {
        self.data.sort_by_key(|(r, c, _)| (*r, *c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_creation() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(0, 1, 1.0);
        coo.push(1, 2, 2.0);
        
        assert_eq!(coo.shape(), (3, 3));
        assert_eq!(coo.nnz(), 2);
    }

    #[test]
    fn test_coo_sort() {
        let mut coo = CooMatrix::<f64>::new(3, 3);
        coo.push(1, 2, 2.0);
        coo.push(0, 1, 1.0);
        coo.sort();
        
        let data = coo.data();
        assert_eq!(data[0].0, 0);
        assert_eq!(data[1].0, 1);
    }
}

