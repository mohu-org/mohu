/// Masked array type and construction functions.
///
/// A `MaskedArray` is a pair of `(data: Buffer, mask: Buffer<bool>)` where
/// `mask[i] == true` means element `i` is **invalid** (masked out).

use mohu_buffer::Buffer;
use mohu_dtype::{DType, Scalar};
use mohu_error::{MohuError, MohuResult};

/// Masked array structure.
///
/// Contains data buffer and boolean mask buffer.
pub struct MaskedArray<T: Scalar> {
    data: Buffer<T>,
    mask: Buffer<bool>,
    fill_value: T,
}

impl<T: Scalar> MaskedArray<T> {
    /// Create a new masked array from data and mask buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if data and mask have different shapes.
    pub fn new(data: Buffer<T>, mask: Buffer<bool>, fill_value: T) -> MohuResult<Self> {
        if data.shape() != mask.shape() {
            return Err(MohuError::ShapeMismatch {
                expected: data.shape().to_vec(),
                got: mask.shape().to_vec(),
            });
        }
        
        Ok(Self {
            data,
            mask,
            fill_value,
        })
    }

    /// Create a masked array from data with no masked elements.
    pub fn from_data(data: Buffer<T>, fill_value: T) -> Self {
        let mask = Buffer::zeros(data.shape(), DType::Bool).unwrap();
        Self {
            data,
            mask,
            fill_value,
        }
    }

    /// Get the data buffer.
    pub fn data(&self) -> &Buffer<T> {
        &self.data
    }

    /// Get the mask buffer.
    pub fn mask(&self) -> &Buffer<bool> {
        &self.mask
    }

    /// Get the fill value.
    pub fn fill_value(&self) -> T {
        self.fill_value
    }

    /// Get the shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the dtype of the data.
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }

    /// Set the fill value.
    pub fn set_fill_value(&mut self, fill_value: T) {
        self.fill_value = fill_value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_array_creation() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false], &[3]).unwrap();
        let masked = MaskedArray::new(data, mask, 0.0).unwrap();
        
        assert_eq!(masked.shape(), &[3]);
        assert_eq!(masked.len(), 3);
    }

    #[test]
    fn test_from_data() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let masked = MaskedArray::from_data(data, 0.0);
        
        assert_eq!(masked.shape(), &[3]);
        assert_eq!(masked.len(), 3);
    }
}

