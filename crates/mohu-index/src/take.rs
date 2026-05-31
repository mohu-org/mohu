use mohu_buffer::buffer::Buffer;
use mohu_buffer::layout::Order;
use mohu_buffer::strides::NdIndexIter;
use mohu_dtype::dispatch_dtype;
use mohu_dtype::dtype::DType;
use mohu_error::{MohuError, MohuResult};

/// Take elements from a buffer along an axis using an array of indices.
///
/// Equivalent to NumPy's `take(a, indices, axis)`.
///
/// # Errors
///
/// - `AxisOutOfRange` if `axis >= src.ndim()`
/// - `DomainError` if indices is not a 1D I64 buffer
/// - `FancyIndexOutOfBounds` if any index is out of range for the axis
pub fn index_take(src: &Buffer, indices: &Buffer, axis: usize) -> MohuResult<Buffer> {
    let ndim = src.ndim();
    if ndim == 0 {
        return Err(MohuError::TooManyIndices {
            given: 1,
            ndim: 0,
        });
    }
    if axis >= ndim {
        return Err(MohuError::AxisOutOfRange {
            axis: axis as i64,
            ndim,
        });
    }
    if indices.ndim() != 1 || indices.dtype() != DType::I64 {
        return Err(MohuError::DomainError {
            op: "index_take",
            reason: "indices must be a 1D I64 array".into(),
        });
    }

    let axis_size = src.shape()[axis];
    let indices_slice = indices.as_slice::<i64>()?;
    let num_indices = indices_slice.len();

    let mut wrapped_indices: Vec<usize> = Vec::with_capacity(num_indices);
    for &idx in indices_slice.iter() {
        let wrapped = if idx < 0 {
            let w = idx + axis_size as i64;
            if w < 0 {
                return Err(MohuError::FancyIndexOutOfBounds {
                    index: idx,
                    axis,
                    size: axis_size,
                });
            }
            w as usize
        } else {
            idx as usize
        };
        if wrapped >= axis_size {
            return Err(MohuError::FancyIndexOutOfBounds {
                index: idx,
                axis,
                size: axis_size,
            });
        }
        wrapped_indices.push(wrapped);
    }

    if num_indices == 0 || src.len() == 0 {
        let mut out_shape: Vec<usize> = src.shape().to_vec();
        out_shape[axis] = 0;
        return Buffer::zeros(src.dtype(), &out_shape);
    }

    // Build output shape
    let mut out_shape: Vec<usize> = src.shape().to_vec();
    out_shape[axis] = num_indices;
    let mut out = Buffer::alloc(src.dtype(), &out_shape, Order::C)?;

    // Build reduced shape (all dims except axis)
    let mut reduced_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != axis {
            reduced_shape.push(src.shape()[i]);
        }
    }

    let mut src_coord = vec![0usize; ndim];
    let mut out_coord = vec![0usize; ndim];

    macro_rules! copy_take {
        ($T:ty) => {
            if ndim == 1 {
                for (pos, &wrapped) in wrapped_indices.iter().enumerate() {
                    let val = src.get::<$T>(&[wrapped])?;
                    out.set::<$T>(&[pos], val)?;
                }
            } else {
                for reduced_coord in NdIndexIter::new(&reduced_shape) {
                    let mut ri = 0;
                    for i in 0..ndim {
                        if i != axis {
                            src_coord[i] = reduced_coord[ri];
                            out_coord[i] = reduced_coord[ri];
                            ri += 1;
                        }
                    }

                    for (pos, &wrapped) in wrapped_indices.iter().enumerate() {
                        src_coord[axis] = wrapped;
                        out_coord[axis] = pos;
                        let val = src.get::<$T>(&src_coord)?;
                        out.set::<$T>(&out_coord, val)?;
                    }
                }
            }
            Ok::<_, MohuError>(())
        };
    }

    dispatch_dtype!(src.dtype(), copy_take)?;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mohu_buffer::buffer::Buffer;

    #[test]
    fn test_index_take_1d() {
        let src = Buffer::from_slice::<i64>(&[10, 20, 30, 40, 50]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[0, 2, 4]).unwrap();
        let result = index_take(&src, &idx, 0).unwrap();
        let expected = Buffer::from_slice::<i64>(&[10, 30, 50]).unwrap();
        assert_eq!(result.as_slice::<i64>().unwrap(), expected.as_slice::<i64>().unwrap());
    }

    #[test]
    fn test_index_take_1d_axis0() {
        let src = Buffer::from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[3, 1]).unwrap();
        let result = index_take(&src, &idx, 0).unwrap();
        assert_eq!(result.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
    }

    #[test]
    fn test_index_take_2d_axis0() {
        let data = &[&[1i64, 2], &[3, 4], &[5, 6]];
        let src = Buffer::from_slice_2d::<i64>(data).unwrap();
        let idx = Buffer::from_slice::<i64>(&[0, 2]).unwrap();
        let result = index_take(&src, &idx, 0).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i64>().unwrap(), &[1, 2, 5, 6]);
    }

    #[test]
    fn test_index_take_2d_axis1() {
        let data = &[&[1i64, 2, 3], &[4, 5, 6]];
        let src = Buffer::from_slice_2d::<i64>(data).unwrap();
        let idx = Buffer::from_slice::<i64>(&[2, 0]).unwrap();
        let result = index_take(&src, &idx, 1).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice::<i64>().unwrap(), &[3, 1, 6, 4]);
    }

    #[test]
    fn test_index_take_negative_indices() {
        let src = Buffer::from_slice::<i64>(&[10, 20, 30]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[-1, -2]).unwrap();
        let result = index_take(&src, &idx, 0).unwrap();
        assert_eq!(result.as_slice::<i64>().unwrap(), &[30, 20]);
    }

    #[test]
    fn test_index_take_empty_indices() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[]).unwrap();
        let result = index_take(&src, &idx, 0).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_index_take_out_of_bounds() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[5]).unwrap();
        assert!(index_take(&src, &idx, 0).is_err());
    }

    #[test]
    fn test_index_take_negative_out_of_bounds() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[-5]).unwrap();
        assert!(index_take(&src, &idx, 0).is_err());
    }

    #[test]
    fn test_index_take_wrong_dtype() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let idx = Buffer::from_slice::<f64>(&[1.0]).unwrap();
        assert!(index_take(&src, &idx, 0).is_err());
    }

    #[test]
    fn test_index_take_bad_axis() {
        let src = Buffer::from_slice::<i64>(&[1, 2, 3]).unwrap();
        let idx = Buffer::from_slice::<i64>(&[0]).unwrap();
        assert!(index_take(&src, &idx, 1).is_err());
    }
}
