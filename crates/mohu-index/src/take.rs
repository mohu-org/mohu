use mohu_buffer::{
    layout::Order,
    strides::{ravel_multi_index, NdIndexIter},
    Buffer,
};
use mohu_dtype::dtype::DType;
use mohu_error::{MohuError, MohuResult};

/// Selects elements along `axis` at positions given by `indices`.
pub fn index_take(src: &Buffer, indices: &Buffer, axis: usize) -> MohuResult<Buffer> {
    if indices.dtype() != DType::I64 {
        return Err(MohuError::domain("index_take", "indices must be i64"));
    }
    if axis >= src.ndim() {
        return Err(MohuError::AxisOutOfRange {
            axis: axis as i64,
            ndim: src.ndim(),
            valid: format!("0..{}", src.ndim()),
        });
    }

    let indices_contig = indices.to_contiguous()?;
    let indices_slice = indices_contig.as_slice::<i64>()?;

    let axis_size = src.shape()[axis] as i64;
    for &idx in indices_slice {
        if idx < 0 || idx >= axis_size {
            return Err(MohuError::IndexOutOfBounds {
                index: idx,
                axis,
                size: src.shape()[axis],
            });
        }
    }

    let mut out_shape = Vec::with_capacity(src.ndim() - 1 + indices.ndim());
    out_shape.extend_from_slice(&src.shape()[..axis]);
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&src.shape()[axis + 1..]);

    let mut out = Buffer::alloc(src.dtype(), &out_shape, Order::C)?;
    let out_ptr = unsafe { out.as_mut_ptr() };

    let src_contig = src.to_contiguous()?;
    let src_ptr = src_contig.as_ptr();
    let itemsize = src_contig.itemsize();

    let out_iter = NdIndexIter::new(&out_shape);
    for (out_flat_idx, out_coord) in out_iter.enumerate() {
        let mut src_coord = Vec::with_capacity(src.ndim());
        src_coord.extend_from_slice(&out_coord[..axis]);

        let indices_coord = &out_coord[axis..axis + indices.ndim()];
        let indices_flat_idx = ravel_multi_index(indices_coord, indices.shape())?;
        let src_axis_idx = indices_slice[indices_flat_idx] as usize;

        src_coord.push(src_axis_idx);
        src_coord.extend_from_slice(&out_coord[axis + indices.ndim()..]);

        let src_flat_idx = ravel_multi_index(&src_coord, src.shape())?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.add(src_flat_idx * itemsize),
                out_ptr.add(out_flat_idx * itemsize),
                itemsize,
            );
        }
    }

    Ok(out)
}
