use std::fmt;

use crate::{MohuError, MohuResult};

/// Extension trait that adds `.context()` and `.with_context()` to any
/// `Result<T, MohuError>`, similar to `anyhow::Context` but typed.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuResult, MohuError, context::ResultExt};
/// fn load(path: &str) -> MohuResult<Vec<u8>> {
///     std::fs::read(path)
///         .map_err(MohuError::from)
///         .context("loading weights file")
/// }
///
/// fn parse(bytes: &[u8]) -> MohuResult<()> {
///     if bytes.is_empty() {
///         return Err(MohuError::Internal("empty bytes".into()));
///     }
///     Ok(())
/// }
///
/// fn run(path: &str) -> MohuResult<()> {
///     let data = load(path)?;
///     parse(&data).with_context(|| format!("parsing {path}"))?;
///     Ok(())
/// }
/// ```
pub trait ResultExt<T>: Sized {
    /// Wraps the `Err` variant with a static context string.
    ///
    /// Prefer this over `with_context` when the context string does not
    /// need to capture any runtime values (avoids a closure allocation).
    fn context(self, ctx: impl fmt::Display + Send + Sync + 'static) -> MohuResult<T>;

    /// Wraps the `Err` variant with a lazily-evaluated context string.
    ///
    /// The closure is only called if the result is `Err`, so it is safe to
    /// put allocation or formatting inside `f` without paying cost on the
    /// happy path.
    fn with_context<C, F>(self, f: F) -> MohuResult<T>
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;

    /// Annotates the error with the name of the operation being performed.
    ///
    /// Equivalent to `.context(op)` but signals intent more clearly at the
    /// call site: `result.op("reshape")` vs `result.context("reshape")`.
    fn op(self, name: &'static str) -> MohuResult<T>;

    /// Annotates the error with an axis number.
    ///
    /// Useful when iterating over axes and the error should include which
    /// axis caused the failure.
    fn axis(self, ax: usize) -> MohuResult<T>;

    /// Promotes `None` to a typed `MohuError`.
    ///
    /// This is only meaningful when called on `Option<T>` via the blanket
    /// impl below.  On `Result` it is equivalent to `.context(ctx)`.
    fn ok_or_mohu(self, err: MohuError) -> MohuResult<T>;
}

// ─── impl for Result<T, MohuError> ───────────────────────────────────────────

impl<T> ResultExt<T> for MohuResult<T> {
    #[inline]
    fn context(self, ctx: impl fmt::Display + Send + Sync + 'static) -> MohuResult<T> {
        self.map_err(|e| MohuError::Context {
            context: ctx.to_string(),
            source: Box::new(e),
        })
    }

    #[inline]
    fn with_context<C, F>(self, f: F) -> MohuResult<T>
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.map_err(|e| MohuError::Context {
            context: f().to_string(),
            source: Box::new(e),
        })
    }

    #[inline]
    fn op(self, name: &'static str) -> MohuResult<T> {
        self.map_err(|e| MohuError::Context {
            context: format!("operation '{name}'"),
            source: Box::new(e),
        })
    }

    #[inline]
    fn axis(self, ax: usize) -> MohuResult<T> {
        self.map_err(|e| MohuError::Context {
            context: format!("axis {ax}"),
            source: Box::new(e),
        })
    }

    #[inline]
    fn ok_or_mohu(self, _err: MohuError) -> MohuResult<T> {
        // For Result, this is a no-op passthrough — the existing error is kept.
        self
    }
}

// ─── impl for Option<T> ───────────────────────────────────────────────────────

impl<T> ResultExt<T> for Option<T> {
    #[inline]
    fn context(self, ctx: impl fmt::Display + Send + Sync + 'static) -> MohuResult<T> {
        self.ok_or_else(|| MohuError::Internal(ctx.to_string()))
    }

    #[inline]
    fn with_context<C, F>(self, f: F) -> MohuResult<T>
    where
        C: fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.ok_or_else(|| MohuError::Internal(f().to_string()))
    }

    #[inline]
    fn op(self, name: &'static str) -> MohuResult<T> {
        self.ok_or_else(|| MohuError::Internal(format!("unexpected None in operation '{name}'")))
    }

    #[inline]
    fn axis(self, ax: usize) -> MohuResult<T> {
        self.ok_or_else(|| MohuError::Internal(format!("unexpected None on axis {ax}")))
    }

    #[inline]
    fn ok_or_mohu(self, err: MohuError) -> MohuResult<T> {
        self.ok_or(err)
    }
}
