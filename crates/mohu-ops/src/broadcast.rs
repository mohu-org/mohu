use mohu_core::mohu_buffer::Buffer;
use mohu_core::mohu_error::{MohuError, MohuResult};

/// Computes the broadcasted output shape for two input shapes.
pub(crate) fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> MohuResult<Vec<usize>> {
    let out_ndim = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(out_ndim);

    for i in 0..out_ndim {
        let l = if i < lhs.len() { lhs[lhs.len() - 1 - i] } else { 1 };
        let r = if i < rhs.len() { rhs[rhs.len() - 1 - i] } else { 1 };
        if l == r || l == 1 || r == 1 {
            out.push(l.max(r));
        } else {
            return Err(MohuError::ShapeMismatch {
                expected: lhs.to_vec(),
                got:      rhs.to_vec(),
            });
        }
    }

    out.reverse();
    Ok(out)
}

pub(crate) fn broadcast_binary_inputs(
    lhs: &Buffer,
    rhs: &Buffer,
) -> MohuResult<(Buffer, Buffer, Vec<usize>)> {
    let out_shape = broadcast_shape(lhs.shape(), rhs.shape())?;

    let lhs_same_shape = lhs.shape() == out_shape.as_slice();
    let rhs_same_shape = rhs.shape() == out_shape.as_slice();

    let lhs_view = if lhs_same_shape {
        lhs.share()
    } else {
        lhs.broadcast_to(&out_shape)?
    };
    let rhs_view = if rhs_same_shape {
        rhs.share()
    } else {
        rhs.broadcast_to(&out_shape)?
    };

    let lhs_contig = if lhs_same_shape && lhs_view.is_c_contiguous() {
        lhs_view
    } else {
        lhs_view.to_contiguous()?
    };
    let rhs_contig = if rhs_same_shape && rhs_view.is_c_contiguous() {
        rhs_view
    } else {
        rhs_view.to_contiguous()?
    };

    Ok((lhs_contig, rhs_contig, out_shape))
}