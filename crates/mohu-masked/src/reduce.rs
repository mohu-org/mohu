/// Reduction operations that skip masked elements.
///
/// Implements sum, mean, min, max, std that ignore masked values.

use super::array::MaskedArray;
use mohu_dtype::Scalar;

/// Sum of non-masked elements.
pub fn sum<T: Scalar>(a: &MaskedArray<T>) -> T {
    let data = a.data();
    let mask = a.mask();
    
    let mut result = T::zero();
    let mut count = 0;
    
    for i in 0..a.len() {
        if !mask.get(i).unwrap_or(&true) {
            result = result + data.get(i).unwrap();
            count += 1;
        }
    }
    
    result
}

/// Mean of non-masked elements.
pub fn mean<T: Scalar>(a: &MaskedArray<T>) -> T {
    let data = a.data();
    let mask = a.mask();
    
    let mut sum = T::zero();
    let mut count = 0;
    
    for i in 0..a.len() {
        if !mask.get(i).unwrap_or(&true) {
            sum = sum + data.get(i).unwrap();
            count += 1;
        }
    }
    
    if count == 0 {
        return a.fill_value();
    }
    
    sum / T::from_usize(count)
}

/// Minimum of non-masked elements.
pub fn min<T: Scalar>(a: &MaskedArray<T>) -> T {
    let data = a.data();
    let mask = a.mask();
    
    let mut result = T::infinity();
    let mut found = false;
    
    for i in 0..a.len() {
        if !mask.get(i).unwrap_or(&true) {
            let val = data.get(i).unwrap();
            if !found || val < result {
                result = val;
                found = true;
            }
        }
    }
    
    if !found {
        a.fill_value()
    } else {
        result
    }
}

/// Maximum of non-masked elements.
pub fn max<T: Scalar>(a: &MaskedArray<T>) -> T {
    let data = a.data();
    let mask = a.mask();
    
    let mut result = T::neg_infinity();
    let mut found = false;
    
    for i in 0..a.len() {
        if !mask.get(i).unwrap_or(&true) {
            let val = data.get(i).unwrap();
            if !found || val > result {
                result = val;
                found = true;
            }
        }
    }
    
    if !found {
        a.fill_value()
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mohu_buffer::Buffer;

    #[test]
    fn test_sum() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false, true], &[4]).unwrap();
        let a = MaskedArray::new(data, mask, 0.0).unwrap();
        
        let result = sum(&a);
        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false, true], &[4]).unwrap();
        let a = MaskedArray::new(data, mask, 0.0).unwrap();
        
        let result = mean(&a);
        assert!((result - 2.0).abs() < 1e-10);
    }
}

