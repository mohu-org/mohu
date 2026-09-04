use mohu_dtype::dtype::DType;
use mohu_random::Rng;

#[test]
fn seeded_streams_reproduce_all_distributions() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    assert_eq!(
        a.uniform(&[8], -2.0, 3.0).unwrap().to_vec::<f64>().unwrap(),
        b.uniform(&[8], -2.0, 3.0).unwrap().to_vec::<f64>().unwrap()
    );
    assert_eq!(
        a.normal(&[8], 1.0, 2.0).unwrap().to_vec::<f64>().unwrap(),
        b.normal(&[8], 1.0, 2.0).unwrap().to_vec::<f64>().unwrap()
    );
    assert_eq!(
        a.integers(&[8], -10, 10).unwrap().to_vec::<i64>().unwrap(),
        b.integers(&[8], -10, 10).unwrap().to_vec::<i64>().unwrap()
    );
}

#[test]
fn uniform_and_normal_validate_parameters() {
    let mut rng = Rng::new(1);
    assert!(rng.uniform(&[1], 1.0, 1.0).is_err());
    assert!(rng.uniform(&[1], f64::NEG_INFINITY, 1.0).is_err());
    assert!(rng.normal(&[1], 0.0, 0.0).is_err());
    assert!(rng.normal(&[1], 0.0, f64::INFINITY).is_err());
}

#[test]
fn normal_has_no_non_finite_values() {
    let mut rng = Rng::new(7);
    let values = rng
        .normal(&[1000], 0.0, 1.0)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert!(values.iter().all(|x| x.is_finite()));
}

#[test]
fn integers_cover_signed_wide_ranges_without_overflow() {
    let mut rng = Rng::new(9);
    let values = rng.integers(&[1000], i64::MIN, i64::MAX).unwrap();
    assert_eq!(values.dtype(), DType::I64);
    assert!(
        values
            .to_vec::<i64>()
            .unwrap()
            .iter()
            .all(|&x| (i64::MIN..i64::MAX).contains(&x))
    );
    assert!(rng.integers(&[1], i64::MAX, i64::MIN).is_err());
}

#[test]
fn zero_sized_shapes_do_not_consume_or_fail() {
    let mut a = Rng::new(5);
    let mut b = Rng::new(5);
    assert_eq!(a.uniform(&[0, 3], 0.0, 1.0).unwrap().len(), 0);
    assert_eq!(
        a.uniform(&[1], 0.0, 1.0).unwrap().to_vec::<f64>().unwrap(),
        b.uniform(&[1], 0.0, 1.0).unwrap().to_vec::<f64>().unwrap()
    );
}

#[test]
fn uniform_is_half_open_and_different_seeds_diverge() {
    let mut a = Rng::new(1);
    let mut b = Rng::new(2);
    let values = a
        .uniform(&[1000], -2.0, 3.0)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert!(values.iter().all(|&x| (-2.0..3.0).contains(&x)));
    assert_ne!(
        values,
        b.uniform(&[1000], -2.0, 3.0)
            .unwrap()
            .to_vec::<f64>()
            .unwrap()
    );
}

#[test]
fn fixed_seed_normal_moments_are_reasonable() {
    let mut rng = Rng::new(123);
    let values = rng
        .normal(&[10_000], 2.0, 3.0)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    assert!((mean - 2.0).abs() < 0.1, "mean={mean}");
    assert!(
        (variance.sqrt() - 3.0).abs() < 0.1,
        "std={}",
        variance.sqrt()
    );
}
