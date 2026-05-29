/// Fill operations for masked arrays.
///
/// Implements the `filled` function to replace masked values with fill_value.

use super::array::MaskedArray;
use mohu_buffer::Buffer;
use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Return a copy of the data with masked values replaced by fill_value.
pub fn filled<T: Scalar>(a: &MaskedArray<T>) -> MohuResult<Buffer<T>> {
    let data = a.data();
    let mask = a.mask();
    let fill_value = a.fill_value();

    let mut result = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        let is_masked = mask.get(i).unwrap_or(&true);
        if *is_masked {
            result.push(fill_value);
        } else {
            result.push(data.get(i)?);
        }
    }

    Buffer::from_slice(&result, a.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filled() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false], &[3]).unwrap();
        let a = MaskedArray::new(data, mask, 999.0).unwrap();

        let result = filled(&a).unwrap();
        let result_vec = result.to_vec().unwrap();
        
        assert_eq!(result_vec, vec![1.0, 999.0, 3.0]);
    }
}

