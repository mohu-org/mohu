use mohu_dtype::DType;

#[test]
fn iter_floats_matches_predicate() {
    let floats: Vec<_> = DType::iter_floats().collect();
    assert_eq!(floats.len(), 4);
    assert!(floats.iter().all(|dtype| dtype.is_float()));
}

#[test]
fn iter_integers_matches_predicate() {
    let integers: Vec<_> = DType::iter_integers().collect();
    assert_eq!(integers.len(), 8);
    assert!(integers.iter().all(|dtype| dtype.is_integer()));
}

#[test]
fn iter_complex_matches_predicate() {
    let complex: Vec<_> = DType::iter_complex().collect();
    assert_eq!(complex.len(), 2);
    assert!(complex.iter().all(|dtype| dtype.is_complex()));
}

#[test]
fn iter_signed_and_unsigned_partition_integers() {
    let signed: Vec<_> = DType::iter_signed().collect();
    let unsigned: Vec<_> = DType::iter_unsigned().collect();
    assert_eq!(signed.len(), 4);
    assert_eq!(unsigned.len(), 4);
    assert!(signed.iter().all(|dtype| dtype.is_signed_integer()));
    assert!(unsigned.iter().all(|dtype| dtype.is_unsigned_integer()));
}
