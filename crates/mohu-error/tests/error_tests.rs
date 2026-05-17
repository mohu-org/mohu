use mohu_error::MohuError;

#[test]
fn is_oom_detects_allocation_failed() {
    let err = MohuError::alloc(1024);
    assert!(err.is_oom());
    assert!(err.is_transient());
    assert!(!err.is_io());
    assert!(!err.is_dlpack());
}

#[test]
fn is_io_detects_io_errors() {
    let err = MohuError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "missing file",
    ));
    assert!(err.is_io());
    assert!(err.is_transient());
    assert!(!err.is_oom());
}

#[test]
fn is_dlpack_detects_dlpack_errors() {
    let err = MohuError::DLPackNullPointer;
    assert!(err.is_dlpack());
    assert!(!err.is_oom());
    assert!(!err.is_io());
}
