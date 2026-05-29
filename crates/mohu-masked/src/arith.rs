/// Arithmetic operations with mask propagation.
///
/// Implements element-wise arithmetic that automatically propagates masks.

use super::array::MaskedArray;
use mohu_buffer::Buffer;
use mohu_dtype::Scalar;
use mohu_error::{MohuError, MohuResult};

/// Add two masked arrays with mask propagation.
pub fn add<T: Scalar>(a: &MaskedArray<T>, b: &MaskedArray<T>) -> MohuResult<MaskedArray<T>> {
    if a.shape() != b.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let data_a = a.data();
    let data_b = b.data();
    let mask_a = a.mask();
    let mask_b = b.mask();

    let mut result_data = Vec::with_capacity(a.len());
    let mut result_mask = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        result_data.push(data_a.get(i)? + data_b.get(i)?);
        result_mask.push(mask_a.get(i)? || mask_b.get(i)?);
    }

    let data_buffer = Buffer::from_slice(&result_data, a.shape())?;
    let mask_buffer = Buffer::from_slice(&result_mask, a.shape())?;

    MaskedArray::new(data_buffer, mask_buffer, a.fill_value())
}

/// Subtract two masked arrays with mask propagation.
pub fn sub<T: Scalar>(a: &MaskedArray<T>, b: &MaskedArray<T>) -> MohuResult<MaskedArray<T>> {
    if a.shape() != b.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let data_a = a.data();
    let data_b = b.data();
    let mask_a = a.mask();
    let mask_b = b.mask();

    let mut result_data = Vec::with_capacity(a.len());
    let mut result_mask = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        result_data.push(data_a.get(i)? - data_b.get(i)?);
        result_mask.push(mask_a.get(i)? || mask_b.get(i)?);
    }

    let data_buffer = Buffer::from_slice(&result_data, a.shape())?;
    let mask_buffer = Buffer::from_slice(&result_mask, a.shape())?;

    MaskedArray::new(data_buffer, mask_buffer, a.fill_value())
}

/// Multiply two masked arrays with mask propagation.
pub fn mul<T: Scalar>(a: &MaskedArray<T>, b: &MaskedArray<T>) -> MohuResult<MaskedArray<T>> {
    if a.shape() != b.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let data_a = a.data();
    let data_b = b.data();
    let mask_a = a.mask();
    let mask_b = b.mask();

    let mut result_data = Vec::with_capacity(a.len());
    let mut result_mask = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        result_data.push(data_a.get(i)? * data_b.get(i)?);
        result_mask.push(mask_a.get(i)? || mask_b.get(i)?);
    }

    let data_buffer = Buffer::from_slice(&result_data, a.shape())?;
    let mask_buffer = Buffer::from_slice(&result_mask, a.shape())?;

    MaskedArray::new(data_buffer, mask_buffer, a.fill_value())
}

/// Divide two masked arrays with mask propagation.
pub fn div<T: Scalar>(a: &MaskedArray<T>, b: &MaskedArray<T>) -> MohuResult<MaskedArray<T>> {
    if a.shape() != b.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let data_a = a.data();
    let data_b = b.data();
    let mask_a = a.mask();
    let mask_b = b.mask();

    let mut result_data = Vec::with_capacity(a.len());
    let mut result_mask = Vec::with_capacity(a.len());

    for i in 0..a.len() {
        result_data.push(data_a.get(i)? / data_b.get(i)?);
        result_mask.push(mask_a.get(i)? || mask_b.get(i)?);
    }

    let data_buffer = Buffer::from_slice(&result_data, a.shape())?;
    let mask_buffer = Buffer::from_slice(&result_mask, a.shape())?;

    MaskedArray::new(data_buffer, mask_buffer, a.fill_value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let data_a = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let mask_a = Buffer::from_slice(&[false, true, false], &[3]).unwrap();
        let a = MaskedArray::new(data_a, mask_a, 0.0).unwrap();

        let data_b = Buffer::from_slice(&[4.0, 5.0, 6.0], &[3]).unwrap();
        let mask_b = Buffer::from_slice(&[false, false, true], &[3]).unwrap();
        let b = MaskedArray::new(data_b, mask_b, 0.0).unwrap();

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape(), &[3]);
    }
}

