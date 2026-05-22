use mohu_buffer::{Buffer, layout::Order};
use mohu_dtype::dtype::DType;
use mohu_error::{MohuError, MohuResult};

/// Extracts elements from `src` corresponding to `true` values in `mask`.
pub fn index_bool(src: &Buffer, mask: &Buffer) -> MohuResult<Buffer> {
    if mask.dtype() != DType::Bool {
        return Err(MohuError::domain("index_bool", "mask must be bool"));
    }
    if mask.shape() != src.shape() {
        return Err(MohuError::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }

    let mask_contig = mask.to_contiguous()?;
    let mask_slice = mask_contig.as_slice::<bool>()?;

    let src_contig = src.to_contiguous()?;
    let itemsize = src_contig.itemsize();
    let src_ptr = src_contig.as_ptr();

    let count = mask_slice.iter().filter(|&&b| b).count();

    let mut out = Buffer::alloc(src.dtype(), &[count], Order::C)?;
    let out_ptr = unsafe { out.as_mut_ptr() };

    let mut out_idx = 0;
    for (i, &b) in mask_slice.iter().enumerate() {
        if b {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(i * itemsize),
                    out_ptr.add(out_idx * itemsize),
                    itemsize,
                );
            }
            out_idx += 1;
        }
    }

    Ok(out)
}
