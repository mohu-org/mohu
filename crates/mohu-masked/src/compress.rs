/// Compression operations for masked arrays.
///
/// Implements `compress` and `compressed` to extract non-masked elements.

use super::array::MaskedArray;
use mohu_buffer::Buffer;
use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Return a 1-D array containing all non-masked elements.
pub fn compress<T: Scalar>(a: &MaskedArray<T>) -> MohuResult<Buffer<T>> {
    let data = a.data();
    let mask = a.mask();

    let mut result = Vec::new();

    for i in 0..a.len() {
        if !mask.get(i).unwrap_or(&true) {
            result.push(data.get(i)?);
        }
    }

    let count = result.len();
    if count == 0 {
        return Buffer::zeros(&[0], a.dtype());
    }

    Buffer::from_slice(&result, &[count])
}

/// Return a 1-D array containing all non-masked elements (alias for compress).
pub fn compressed<T: Scalar>(a: &MaskedArray<T>) -> MohuResult<Buffer<T>> {
    compress(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false, true], &[4]).unwrap();
        let a = MaskedArray::new(data, mask, 0.0).unwrap();

        let result = compress(&a).unwrap();
        let result_vec = result.to_vec().unwrap();
        
        assert_eq!(result_vec, vec![1.0, 3.0]);
    }
}

