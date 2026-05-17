use mohu_dtype::DType;

#[test]
fn iter_floats_count_and_predicate() {
    let floats: Vec<_> = DType::iter_floats().collect();
    assert_eq!(floats.len(), 4);
    assert!(floats.iter().all(|d| d.is_float()));
}

#[test]
fn iter_integers_count_and_predicate() {
    let ints: Vec<_> = DType::iter_integers().collect();
    assert_eq!(ints.len(), 8);
    assert!(ints.iter().all(|d| d.is_integer()));
}

#[test]
fn iter_complex_count_and_predicate() {
    let cx: Vec<_> = DType::iter_complex().collect();
    assert_eq!(cx.len(), 2);
    assert!(cx.iter().all(|d| d.is_complex()));
}

#[test]
fn iter_signed_and_unsigned_partition_integers() {
    let signed: Vec<_> = DType::iter_signed().collect();
    let unsigned: Vec<_> = DType::iter_unsigned().collect();
    assert_eq!(signed.len(), 4);
    assert_eq!(unsigned.len(), 4);
    assert!(signed.iter().all(|d| d.is_signed_integer()));
    assert!(unsigned.iter().all(|d| d.is_unsigned_integer()));
    assert_eq!(signed.len() + unsigned.len(), DType::iter_integers().count());
}
