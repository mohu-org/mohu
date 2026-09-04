//! Integer-array take indexing.
use mohu_buffer::{Buffer, NdIndexIter, Order};
use mohu_dtype::{dispatch_dtype, dtype::DType};
use mohu_error::{MohuError, MohuResult};

/// Return an owning buffer selected along `axis` by non-negative I64 indices.
pub fn index_take(src: &Buffer, indices: &Buffer, axis: usize) -> MohuResult<Buffer> {
    if indices.dtype() != DType::I64 {
        return Err(MohuError::DomainError {
            op: "index_take",
            reason: format!("indices must have I64 dtype, got {}", indices.dtype()),
        });
    }
    if axis >= src.ndim() {
        return Err(MohuError::AxisOutOfRange {
            axis: axis as i64,
            ndim: src.ndim(),
            valid: if src.ndim() == 0 {
                "none".into()
            } else {
                format!("0..{}", src.ndim())
            },
        });
    }
    let axis_size = src.shape()[axis];
    let mut normalized = Vec::with_capacity(indices.len());
    for coord in NdIndexIter::new(indices.shape()) {
        let index = indices.get::<i64>(&coord)?;
        if index < 0 || (index as u128) >= axis_size as u128 {
            return Err(MohuError::IndexOutOfBounds {
                index,
                axis,
                size: axis_size,
            });
        }
        normalized.push(index as usize);
    }
    let mut out_shape = Vec::with_capacity(src.ndim() - 1 + indices.ndim());
    out_shape.extend_from_slice(&src.shape()[..axis]);
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&src.shape()[axis + 1..]);
    let mut output = Buffer::alloc(src.dtype(), &out_shape, Order::C)?;
    macro_rules! copy {
        ($ty:ty) => {{
            for out_coord in NdIndexIter::new(&out_shape) {
                let idx_coord = &out_coord[axis..axis + indices.ndim()];
                let selected = indices.get::<i64>(idx_coord)? as usize;
                let mut src_coord = Vec::with_capacity(src.ndim());
                src_coord.extend_from_slice(&out_coord[..axis]);
                src_coord.push(selected);
                src_coord.extend_from_slice(&out_coord[axis + indices.ndim()..]);
                let value = src.get::<$ty>(&src_coord)?;
                output.set::<$ty>(&out_coord, value)?;
            }
            Ok(output)
        }};
    }
    let _ = normalized;
    dispatch_dtype!(src.dtype(), copy)
}
