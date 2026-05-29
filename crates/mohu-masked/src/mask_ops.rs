/// Mask operations for masked arrays.
///
/// Implements masked_where, masked_equal, getmask, getdata.

use super::array::MaskedArray;
use mohu_buffer::Buffer;
use mohu_dtype::{DType, Scalar};
use mohu_error::MohuResult;

/// Mask array where condition is true.
pub fn masked_where<T: Scalar>(condition: &Buffer<bool>, a: &Buffer<T>, fill_value: T) -> MohuResult<MaskedArray<T>> {
    let data = a.clone();
    let mask = condition.clone();
    MaskedArray::new(data, mask, fill_value)
}

/// Mask array where elements are equal to value.
pub fn masked_equal<T: Scalar>(a: &Buffer<T>, value: T, fill_value: T) -> MohuResult<MaskedArray<T>> {
    let data = a.clone();
    let mut mask = Vec::with_capacity(a.len());
    
    for i in 0..a.len() {
        mask.push(a.get(i)? == value);
    }
    
    let mask_buffer = Buffer::from_slice(&mask, a.shape())?;
    MaskedArray::new(data, mask_buffer, fill_value)
}

/// Get the mask from a masked array.
pub fn getmask<T: Scalar>(a: &MaskedArray<T>) -> &Buffer<bool> {
    a.mask()
}

/// Get the data from a masked array.
pub fn getdata<T: Scalar>(a: &MaskedArray<T>) -> &Buffer<T> {
    a.data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_where() {
        let condition = Buffer::from_slice(&[true, false, true], &[3]).unwrap();
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let masked = masked_where(&condition, &data, 0.0).unwrap();
        
        let mask_vec = masked.mask().to_vec().unwrap();
        assert_eq!(mask_vec, vec![true, false, true]);
    }

    #[test]
    fn test_masked_equal() {
        let data = Buffer::from_slice(&[1.0, 2.0, 2.0, 3.0], &[4]).unwrap();
        let masked = masked_equal(&data, 2.0, 0.0).unwrap();
        
        let mask_vec = masked.mask().to_vec().unwrap();
        assert_eq!(mask_vec, vec![false, true, true, false]);
    }
}

