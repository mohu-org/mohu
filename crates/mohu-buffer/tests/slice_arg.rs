use mohu_buffer::SliceArg;

#[test]
fn from_usize_range() {
    let arg: SliceArg = (2..5).into();
    assert_eq!(arg.start, Some(2));
    assert_eq!(arg.stop, Some(5));
    assert_eq!(arg.step, None);

    let (start, count, step) = arg.resolve(10).unwrap();
    assert_eq!((start, count, step), (2, 3, 1));
}

#[test]
fn from_range_full() {
    let arg: SliceArg = (..).into();
    assert_eq!(arg, SliceArg::FULL);

    let (start, count, step) = arg.resolve(8).unwrap();
    assert_eq!((start, count, step), (0, 8, 1));
}
