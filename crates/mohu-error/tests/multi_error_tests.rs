use mohu_error::{MohuError, MultiError};

fn error(message: &'static str) -> MohuError {
    MohuError::domain("test", message)
}

#[test]
fn empty_collector_is_ok() {
    assert!(MultiError::new().into_result().is_ok());
}

#[test]
fn one_error_is_returned_directly() {
    let mut errors = MultiError::new();
    errors.push(error("first"));
    let result = errors.into_result();
    assert!(
        matches!(result, Err(MohuError::DomainError { op: "test", ref reason }) if reason == "first")
    );
}

#[test]
fn multiple_errors_are_wrapped() {
    let mut errors = MultiError::new();
    errors.push(error("first"));
    errors.push(error("second"));
    assert!(matches!(errors.into_result(), Err(MohuError::Multiple(_))));
}

#[test]
fn iteration_preserves_insertion_order() {
    let mut errors = MultiError::new();
    errors.push(error("A"));
    errors.push(error("B"));
    errors.push(error("C"));
    let messages: Vec<_> = errors.iter().map(ToString::to_string).collect();
    assert!(messages[0].contains("A"));
    assert!(messages[1].contains("B"));
    assert!(messages[2].contains("C"));
}

#[test]
fn display_contains_count_and_messages() {
    let mut errors = MultiError::new();
    errors.push(error("left"));
    errors.push(error("right"));
    let text = errors.to_string();
    assert!(text.contains("2 error(s)"));
    assert!(text.contains("left"));
    assert!(text.contains("right"));
}

#[test]
fn collect_keeps_only_errors() {
    let mut errors = MultiError::new();
    errors.collect::<()>(Ok(()));
    errors.collect::<()>(Err(error("bad")));
    assert_eq!(errors.len(), 1);
    assert!(errors.has_errors());
}

#[test]
fn borrowed_into_iterator_matches_iter() {
    let mut errors = MultiError::new();
    errors.push(error("A"));
    errors.push(error("B"));
    assert_eq!((&errors).into_iter().count(), 2);
}
