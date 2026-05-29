/// I/O operations for masked arrays.
///
/// Implements serialization/deserialization for masked arrays (NPY extension).

use super::array::MaskedArray;
use mohu_buffer::Buffer;
use mohu_dtype::Scalar;
use mohu_error::MohuResult;

/// Serialize masked array to a simple format (placeholder for NPY extension).
///
/// For now, returns a tuple of (data_bytes, mask_bytes).
pub fn serialize<T: Scalar>(a: &MaskedArray<T>) -> MohuResult<(Vec<u8>, Vec<u8>)> {
    let data_vec = a.data().to_vec()?;
    let mask_vec = a.mask().to_vec()?;
    
    // Convert to bytes (simplified - in production use proper serialization)
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data_vec.as_ptr() as *const u8,
            data_vec.len() * std::mem::size_of::<T>(),
        )
    }.to_vec();
    
    let mask_bytes = unsafe {
        std::slice::from_raw_parts(
            mask_vec.as_ptr() as *const u8,
            mask_vec.len() * std::mem::size_of::<bool>(),
        )
    }.to_vec();
    
    Ok((data_bytes, mask_bytes))
}

/// Deserialize masked array from bytes (placeholder for NPY extension).
pub fn deserialize<T: Scalar>(data_bytes: &[u8], mask_bytes: &[u8], shape: &[usize], fill_value: T) -> MohuResult<MaskedArray<T>> {
    let data_len = data_bytes.len() / std::mem::size_of::<T>();
    let mask_len = mask_bytes.len() / std::mem::size_of::<bool>();
    
    // Convert bytes to Vec<T> and Vec<bool> (simplified)
    let data = unsafe {
        Vec::from_raw_parts(
            data_bytes.as_ptr() as *mut T,
            data_len,
            data_len,
        )
    };
    
    let mask = unsafe {
        Vec::from_raw_parts(
            mask_bytes.as_ptr() as *mut bool,
            mask_len,
            mask_len,
        )
    };
    
    std::mem::forget(data_bytes);
    std::mem::forget(mask_bytes);
    
    let data_buffer = Buffer::from_slice(&data, shape)?;
    let mask_buffer = Buffer::from_slice(&mask, shape)?;
    
    MaskedArray::new(data_buffer, mask_buffer, fill_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize() {
        let data = Buffer::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let mask = Buffer::from_slice(&[false, true, false], &[3]).unwrap();
        let a = MaskedArray::new(data, mask, 0.0).unwrap();

        let (data_bytes, mask_bytes) = serialize(&a).unwrap();
        let result = deserialize(&data_bytes, &mask_bytes, &[3], 0.0).unwrap();
        
        assert_eq!(result.shape(), &[3]);
    }
}

