//! Runnable tour of mohu-error patterns for new contributors.

use mohu_error::{
    codes::ErrorCode, bail, ensure, ErrorKind, MohuError, MohuResult, MultiError, ResultExt,
};

fn basic_result() -> MohuResult<i32> {
    let value: MohuResult<i32> = Ok(42);
    value.map(|n| n * 2)
}

fn early_exit_with_bail(flag: bool) -> MohuResult<()> {
    if flag {
        bail!(MohuError::Internal("simulated failure".into()));
    }
    Ok(())
}

fn precondition_with_ensure(value: f64) -> MohuResult<f64> {
    ensure!(value.is_finite(), MohuError::DomainError {
        op: "sqrt",
        reason: "input must be finite".into(),
    });
    Ok(value.sqrt())
}

fn context_wrapping(path: &str) -> MohuResult<String> {
    std::fs::read_to_string(path)
        .map_err(MohuError::Io)
        .with_context(|| format!("failed to read config at {path}"))
}

fn accumulate_with_multi_error(shapes: &[&[usize]]) -> MohuResult<()> {
    let mut errors = MultiError::new();
    for (axis, shape) in shapes.iter().enumerate() {
        if shape.iter().any(|&dim| dim == 0) {
            errors.push(MohuError::ZeroSizedDimension { axis });
        }
    }
    errors.into_result()
}

fn match_error_code(err: MohuError) {
    match err.code() {
        ErrorCode::ShapeMismatch => println!("shape mismatch: {err}"),
        ErrorCode::DTypeMismatch => println!("dtype mismatch: {err}"),
        other => println!("other error ({other:?}): {err}"),
    }
}

fn coarse_retry_logic(err: &MohuError) {
    match err.kind() {
        ErrorKind::Usage => println!("fix caller input: {err}"),
        ErrorKind::Runtime if err.is_transient() => println!("retry may help: {err}"),
        ErrorKind::System => println!("environment issue: {err}"),
        _ => println!("unexpected: {err}"),
    }
}

fn main() {
    println!("basic: {:?}", basic_result());

    if let Err(err) = early_exit_with_bail(true) {
        println!("bail: {err}");
    }

    println!("ensure: {:?}", precondition_with_ensure(4.0));

    if let Err(err) = context_wrapping("/tmp/does-not-exist-mohu-example.txt") {
        println!("context: {err}");
    }

    if let Err(err) = accumulate_with_multi_error(&[&[2, 0], &[3, 4]]) {
        println!("multi: {err}");
    }

    match_error_code(MohuError::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![2, 4],
    });

    coarse_retry_logic(&MohuError::alloc(1_024));
}
