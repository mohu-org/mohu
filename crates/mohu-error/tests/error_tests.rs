use mohu_error::MohuError;

#[test]
fn is_oom_only_for_allocation_failed() {
    assert!(MohuError::alloc(1024).is_oom());
    assert!(!MohuError::bug("x").is_oom());
    assert!(!MohuError::Io(std::io::Error::other("disk")).is_oom());
}

#[test]
fn is_io_only_for_io_variant() {
    assert!(MohuError::Io(std::io::Error::other("read")).is_io());
    assert!(!MohuError::alloc(1).is_io());
    assert!(!MohuError::ShapeMismatch {
        expected: vec![2],
        got: vec![3],
    }
    .is_io());
}

#[test]
fn is_dlpack_covers_all_dlpack_variants() {
    assert!(MohuError::DLPackNullPointer.is_dlpack());
    assert!(MohuError::DLPackUnsupportedDevice { device_type: 1 }.is_dlpack());
    assert!(MohuError::DLPackVersionMismatch {
        supported_major: 1,
        got_major: 2,
        got_minor: 0,
    }
    .is_dlpack());
    assert!(MohuError::DLPackUnsupportedDType {
        code: 1,
        bits: 32,
        lanes: 1,
    }
    .is_dlpack());
    assert!(MohuError::DLPackInvalid("bad".into()).is_dlpack());
    assert!(!MohuError::bug("x").is_dlpack());
}
