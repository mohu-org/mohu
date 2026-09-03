//! Compile-checks the snippets in this crate's README.md so they cannot rot.

use mohu_error::{MohuError, MohuResult, ResultExt, ensure};

fn safe_divide(a: f64, b: f64) -> MohuResult<f64> {
    ensure!(b != 0.0, MohuError::DivisionByZero);
    Ok(a / b)
}

fn compute() -> MohuResult<f64> {
    safe_divide(1.0, 0.0).context("while normalising the input row")
}

#[test]
fn readme_usage_snippet() {
    assert_eq!(safe_divide(6.0, 2.0).unwrap(), 3.0);
    assert!(compute().is_err());
}
