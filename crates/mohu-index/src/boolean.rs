//! Boolean mask indexing.
use mohu_buffer::{Buffer, NdIndexIter, Order};
use mohu_dtype::{dispatch_dtype, dtype::DType};
use mohu_error::{MohuError, MohuResult};

/// Return a new one-dimensional buffer containing values selected by `mask`.
pub fn index_bool(src: &Buffer, mask: &Buffer) -> MohuResult<Buffer> {
    if mask.dtype() != DType::Bool {
        return Err(MohuError::DomainError {
            op: "index_bool",
            reason: format!("mask must have Bool dtype, got {}", mask.dtype()),
        });
    }
    if mask.shape() != src.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }
    let mut count = 0;
    for idx in NdIndexIter::new(src.shape()) {
        if mask.get::<bool>(&idx)? {
            count += 1;
        }
    }
    let mut output = Buffer::alloc(src.dtype(), &[count], Order::C)?;
    macro_rules! copy {
        ($ty:ty) => {{
            let mut pos = 0;
            for idx in NdIndexIter::new(src.shape()) {
                if mask.get::<bool>(&idx)? {
                    let value = src.get::<$ty>(&idx)?;
                    output.set::<$ty>(&[pos], value)?;
                    pos += 1;
                }
            }
            Ok(output)
        }};
    }
    dispatch_dtype!(src.dtype(), copy)
}
