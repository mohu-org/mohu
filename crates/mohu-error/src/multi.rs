use std::fmt;

use crate::{MohuError, MohuResult};

/// A collector for multiple `MohuError` values.
///
/// `MultiError` lets validation and parsing passes accumulate all errors
/// rather than stopping at the first failure. This is essential for
/// giving users complete feedback — e.g. reporting every invalid column
/// in one call instead of one per call.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, MohuResult, multi::MultiError};
/// fn validate(shapes: &[&[usize]]) -> MohuResult<()> {
///     let mut errs = MultiError::new();
///
///     for (i, &shape) in shapes.iter().enumerate() {
///         if shape.iter().any(|&d| d == 0) {
///             errs.push(MohuError::ZeroSizedDimension { axis: i });
///         }
///     }
///
///     errs.into_result()
/// }
/// ```
#[derive(Debug, Default)]
pub struct MultiError {
    errors: Vec<MohuError>,
}

impl MultiError {
    /// Creates an empty `MultiError` collector.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Creates a `MultiError` pre-allocated for `capacity` errors.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            errors: Vec::with_capacity(capacity),
        }
    }

    /// Adds an error to the collection.
    pub fn push(&mut self, err: MohuError) {
        self.errors.push(err);
    }

    /// Adds an error only if `result` is `Err`, discarding the `Ok` value.
    ///
    /// This is the idiomatic way to run a fallible check and continue:
    ///
    /// ```rust
    /// # use mohu_error::{MohuError, MohuResult, multi::MultiError};
    /// # fn check(_: usize) -> MohuResult<()> { Ok(()) }
    /// let mut errs = MultiError::new();
    /// errs.collect(check(0));
    /// errs.collect(check(1));
    /// errs.into_result()
    /// # .ok();
    /// ```
    pub fn collect<T>(&mut self, result: MohuResult<T>) {
        if let Err(e) = result {
            self.errors.push(e);
        }
    }

    /// Returns `true` if no errors have been accumulated.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// Returns `true` if at least one error has been accumulated.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Returns the number of accumulated errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Returns `true` if no errors have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Iterates over the accumulated errors.
    pub fn iter(&self) -> std::slice::Iter<'_, MohuError> {
        self.errors.iter()
    }

    /// Consumes the collector and returns:
    /// - `Ok(())` if no errors were accumulated
    /// - `Err(MohuError::Multiple(_))` if one or more errors were accumulated
    pub fn into_result(self) -> MohuResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else if self.errors.len() == 1 {
            // Unwrap the single error — no need to box it in Multiple.
            Err(self.errors.into_iter().next().unwrap())
        } else {
            Err(MohuError::Multiple(self))
        }
    }

    /// Consumes the collector and returns the raw `Vec<MohuError>`.
    pub fn into_errors(self) -> Vec<MohuError> {
        self.errors
    }

    /// Merges another `MultiError` into this one.
    pub fn extend_from(&mut self, other: MultiError) {
        self.errors.extend(other.errors);
    }
}

impl fmt::Display for MultiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} error(s):", self.errors.len())?;
        for (i, e) in self.errors.iter().enumerate() {
            write!(f, "\n  [{i}] {e}")?;
        }
        Ok(())
    }
}

impl IntoIterator for MultiError {
    type Item = MohuError;
    type IntoIter = std::vec::IntoIter<MohuError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

impl<'a> IntoIterator for &'a MultiError {
    type Item = &'a MohuError;
    type IntoIter = std::slice::Iter<'a, MohuError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.iter()
    }
}

impl FromIterator<MohuError> for MultiError {
    fn from_iter<I: IntoIterator<Item = MohuError>>(iter: I) -> Self {
        Self {
            errors: iter.into_iter().collect(),
        }
    }
}

impl Extend<MohuError> for MultiError {
    fn extend<I: IntoIterator<Item = MohuError>>(&mut self, iter: I) {
        self.errors.extend(iter);
    }
}
