use crate::{reporter::ErrorReporter, MohuError};

#[test]
fn compact_report_includes_message() {
    let error = MohuError::Internal("compact error".into());

    let result = format!("{}", ErrorReporter::compact(&error));

    assert!(result.contains("compact error"));
}

#[test]
fn full_report_includes_message() {
    let error = MohuError::Internal("full error".into());

    let result = format!("{}", ErrorReporter::full(&error));

    assert!(result.contains("full error"));
}

#[test]
fn json_report_contains_message_field() {
    let error = MohuError::Internal("json error".into());

    let result = format!("{}", ErrorReporter::json(&error));

    assert!(result.contains("\"message\""));
    assert!(result.contains("json error"));
    

}

#[test]
fn severity_returns_fatal_for_internal_error() {
    let error = MohuError::Internal("severity error".into());

    let severity = error.severity().to_string();

    assert_eq!(severity, "fatal");
}