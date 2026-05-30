use mohu_buffer::{AllocStats, Buffer};
use mohu_dtype::DType;

#[test]
fn single_allocation_increments_count_and_bytes() {
    let before = AllocStats::snapshot();

    let _buf = Buffer::zeros(DType::F64, &[100]).unwrap();

    let after = AllocStats::snapshot();

    let alloc_delta = after.alloc_count - before.alloc_count;

    assert_eq!(alloc_delta, 1);

    assert!(after.live_bytes > before.live_bytes);
}

#[test]
fn drop_decrements_live_bytes() {
    let before = AllocStats::snapshot();

    {
        let _buf = Buffer::zeros(DType::F64, &[200]).unwrap();
    }

    let after = AllocStats::snapshot();

    assert!(after.free_count >= before.free_count + 1);

    assert!(after.live_bytes <= before.live_bytes);
}

#[test]
fn clone_increments_alloc_count() {
    let before = AllocStats::snapshot();

    let buf = Buffer::zeros(DType::F64, &[50]).unwrap();
    let _clone = buf.clone();

    let after = AllocStats::snapshot();

    assert_eq!(after.alloc_count, before.alloc_count + 2);
}

#[test]
fn zero_size_allocation() {
    let before = AllocStats::snapshot();

    let _buf = Buffer::zeros(DType::F64, &[]).unwrap();

    let after = AllocStats::snapshot();

    assert!(after.alloc_count >= before.alloc_count);

    assert!(after.live_bytes >= before.live_bytes);
}
