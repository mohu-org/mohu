use mohu_buffer::buffer::Buffer;
use mohu_buffer::strides::NdIndexIter;
use mohu_dtype::dispatch_dtype;
use mohu_dtype::dtype::DType;
use mohu_error::{MohuError, MohuResult};

/// Boolean mask indexing.
///
/// Returns a new 1D buffer containing only the elements of `src` where the
/// corresponding element of `mask` is `true`.
///
/// # Errors
///
/// - `DomainError` if `mask.dtype() != DType::Bool`
/// - `BoolIndexShapeMismatch` if `mask.shape() != src.shape()`
pub fn index_bool(src: &Buffer, mask: &Buffer) -> MohuResult<Buffer> {
    if mask.dtype() != DType::Bool {
        return Err(MohuError::DomainError {
            op: "index_bool",
            reason: "mask dtype must be Bool".into(),
        });
    }
    if mask.shape() != src.shape() {
        return Err(MohuError::BoolIndexShapeMismatch {
            index_shape: mask.shape().to_vec(),
            array_shape: src.shape().to_vec(),
        });
    }

    let mut true_coords: Vec<Vec<usize>> = Vec::new();

    for coord in NdIndexIter::new(src.shape()) {
        if mask.get::<bool>(&coord)? {
            true_coords.push(coord.to_vec());
        }
    }

    let out_len = true_coords.len();
    let mut out = Buffer::zeros(src.dtype(), &[out_len])?;

    if out_len == 0 {
        return Ok(out);
    }

    macro_rules! copy_bool {
        ($T:ty) => {
            for (i, coord) in true_coords.iter().enumerate() {
                let val = src.get::<$T>(coord)?;
                out.set::<$T>(&[i], val)?;
            }
            Ok::<_, MohuError>(())
        };
    }

    dispatch_dtype!(src.dtype(), copy_bool)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mohu_buffer::buffer::Buffer;

    #[test]
    fn test_index_bool_1d() {
        let src = Buffer::from_slice::<i64>(&[10, 20, 30, 40, 50]).unwrap();
        let mask = Buffer::from_slice::<bool>(&[true, false, true, false, true]).unwrap();
        let result = index_bool(&src, &mask).unwrap();
        let expected = Buffer::from_slice::<i64>(&[10, 30, 50]).unwrap();
        assert_eq!(result.as_slice::<i64>().unwrap(), expected.as_slice::<i64>().unwrap());
    }

    #[test]
    fn test_index_bool_all_false() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let mask = Buffer::from_slice::<bool>(&[false, false, false]).unwrap();
        let result = index_bool(&src, &mask).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_index_bool_all_true() {
        let src = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0]).unwrap();
        let mask = Buffer::from_slice::<bool>(&[true, true, true]).unwrap();
        let result = index_bool(&src, &mask).unwrap();
        assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_index_bool_2d() {
        let data = &[&[1i64, 2, 3], &[4, 5, 6]];
        let src = Buffer::from_slice_2d::<i64>(data).unwrap();
        let mask = Buffer::from_slice_2d::<bool>(&[&[true, false, true], &[false, true, false]]).unwrap();
        let result = index_bool(&src, &mask).unwrap();
        let expected = Buffer::from_slice::<i64>(&[1, 3, 5]).unwrap();
        assert_eq!(result.as_slice::<i64>().unwrap(), expected.as_slice::<i64>().unwrap());
    }

    #[test]
    fn test_index_bool_wrong_dtype() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let mask = Buffer::from_slice::<i64>(&[1, 0, 1]).unwrap();
        assert!(index_bool(&src, &mask).is_err());
    }

    #[test]
    fn test_index_bool_shape_mismatch() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let mask = Buffer::from_slice::<bool>(&[true, false]).unwrap();
        assert!(index_bool(&src, &mask).is_err());
    }
}
