/// Rich terminal error reporter — formats a `MohuError` the way rustc
/// formats compile errors: error code prefix, full message, cause chain,
/// and hints on separate lines.
///
/// # Output modes
///
/// - **Compact** (default): single line with error code.
/// - **Full**: multi-line with cause chain and any `hint:` text promoted
///   onto its own indented line.
/// - **JSON**: machine-readable structured output.
///
/// # ANSI colour
///
/// Colour is controlled by the `MOHU_COLOR` environment variable:
/// - unset / `"auto"`: enabled if stdout is a TTY
/// - `"always"`: always emit ANSI codes
/// - `"never"`: never emit ANSI codes
///
/// When compiled for `wasm32` targets TTY detection is not available
/// and colour defaults to off.
///
/// # Example
///
/// ```rust
/// # use mohu_error::{MohuError, context::ResultExt, reporter::ErrorReporter};
/// let err = Err::<(), _>(MohuError::SingularMatrix)
///     .context("computing weight matrix inverse")
///     .context("normalising layer outputs")
///     .unwrap_err();
///
/// // compact
/// eprintln!("{}", ErrorReporter::compact(&err));
///
/// // full
/// eprintln!("{}", ErrorReporter::full(&err));
///
/// // JSON
/// eprintln!("{}", ErrorReporter::json(&err));
/// ```
use std::fmt;

use crate::{chain::ErrorChain, MohuError};

// ─── ANSI escape sequences ────────────────────────────────────────────────────

mod ansi {
    pub const RESET:     &str = "\x1b[0m";
    pub const BOLD:      &str = "\x1b[1m";
    pub const DIM:       &str = "\x1b[2m";
    pub const BOLD_RED:  &str = "\x1b[1;31m";
    pub const BOLD_CYAN: &str = "\x1b[1;36m";
}

// ─── colour detection ─────────────────────────────────────────────────────────

fn colour_enabled() -> bool {
    match std::env::var("MOHU_COLOR").as_deref() {
        Ok("always") => true,
        Ok("never")  => false,
        // "auto" or unset: check for NO_COLOR and TERM
        _ => {
            if std::env::var("NO_COLOR").is_ok() {
                return false;
            }
            // On non-wasm targets, check if stderr is a TTY via a simple
            // heuristic: if the TERM variable is "dumb" or unset, no colour.
            match std::env::var("TERM").as_deref() {
                Ok("dumb") | Err(_) => false,
                Ok(_) => true,
            }
        }
    }
}

// ─── ReportMode ──────────────────────────────────────────────────────────────

/// The display mode for an [`ErrorReporter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportMode {
    /// Single line: `error[E1001] message`.
    Compact,
    /// Multi-line with cause chain and separated hints.
    Full,
    /// Machine-readable JSON object (no ANSI codes).
    Json,
}

// ─── ErrorReporter ───────────────────────────────────────────────────────────

/// Formats a `MohuError` for terminal output.
///
/// Implements `Display` so it can be used directly in `format!`, `eprintln!`,
/// and similar macros.
pub struct ErrorReporter<'a> {
    error: &'a MohuError,
    mode:  ReportMode,
    color: bool,
}

impl<'a> ErrorReporter<'a> {
    /// Creates a new reporter with the given mode.
    pub fn new(error: &'a MohuError, mode: ReportMode) -> Self {
        Self { error, mode, color: colour_enabled() }
    }

    /// Compact single-line reporter.
    pub fn compact(error: &'a MohuError) -> Self {
        Self::new(error, ReportMode::Compact)
    }

    /// Full multi-line reporter with cause chain.
    pub fn full(error: &'a MohuError) -> Self {
        Self::new(error, ReportMode::Full)
    }

    /// JSON reporter (machine-readable, no ANSI).
    pub fn json(error: &'a MohuError) -> Self {
        Self { error, mode: ReportMode::Json, color: false }
    }

    /// Forces colour on or off regardless of environment detection.
    pub fn with_color(mut self, color: bool) -> Self {
        self.color = color;
        self
    }

    // ── internal helpers ──────────────────────────────────────────────────

    fn c<'s>(&self, code: &'s str) -> &'s str {
        if self.color { code } else { "" }
    }

    fn reset(&self) -> &'static str {
        if self.color { ansi::RESET } else { "" }
    }

    fn fmt_compact(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let root = ErrorChain::root(self.error);
        let code = root.code();
        write!(
            f,
            "{bold_red}error[{code}]{reset} {bold}{msg}{reset}",
            bold_red = self.c(ansi::BOLD_RED),
            code      = code,
            reset     = self.reset(),
            bold      = self.c(ansi::BOLD),
            msg       = self.error,
        )
    }

    fn fmt_full(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let root  = ErrorChain::root(self.error);
        let code  = root.code();
        let depth = ErrorChain::depth(self.error);

        // ── header line ───────────────────────────────────────────────────
        writeln!(
            f,
            "{bold_red}error[{code}]{reset}{bold}: {msg}{reset}",
            bold_red = self.c(ansi::BOLD_RED),
            code      = code,
            reset     = self.reset(),
            bold      = self.c(ansi::BOLD),
            msg       = root,
        )?;

        // ── context chain (outermost → innermost) ─────────────────────────
        if depth > 0 {
            let ctxs = ErrorChain::context_messages(self.error);
            writeln!(
                f,
                "{dim}  context chain:{reset}",
                dim   = self.c(ansi::DIM),
                reset = self.reset(),
            )?;
            for (i, ctx) in ctxs.iter().enumerate().rev() {
                writeln!(
                    f,
                    "  {dim}{arrow}{reset} {ctx}",
                    dim   = self.c(ansi::DIM),
                    arrow = if i == 0 { "└─" } else { "├─" },
                    reset = self.reset(),
                    ctx   = ctx,
                )?;
            }
        }

        // ── hint lines (extracted from the Display message) ───────────────
        let root_msg = root.to_string();
        for line in root_msg.lines().skip(1) {
            if let Some(hint) = line.strip_prefix("hint: ") {
                writeln!(
                    f,
                    "  {cyan}hint{reset}: {hint}",
                    cyan  = self.c(ansi::BOLD_CYAN),
                    reset = self.reset(),
                    hint  = hint,
                )?;
            }
        }

        // ── error code description ────────────────────────────────────────
        writeln!(
            f,
            "  {dim}[{code}] {domain} error{reset}",
            dim    = self.c(ansi::DIM),
            code   = code,
            domain = code.domain(),
            reset  = self.reset(),
        )?;

        Ok(())
    }

    fn fmt_json(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let root  = ErrorChain::root(self.error);
        let code  = root.code() as u32;
        let kind  = crate::kind::ErrorKind::from(root.code());
        let depth = ErrorChain::depth(self.error);

        // Build context array
        let ctxs  = ErrorChain::context_messages(self.error);
        let ctx_json: Vec<String> = ctxs.iter()
            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
            .collect();

        // Extract first line of root error as the primary message,
        // and subsequent "hint: ..." lines separately.
        let root_msg = root.to_string();
        let mut lines = root_msg.lines();
        let primary = lines.next().unwrap_or("").replace('"', "\\\"");
        let hints: Vec<String> = lines
            .filter_map(|l| l.strip_prefix("hint: "))
            .map(|h| format!("\"{}\"", h.replace('"', "\\\"")))
            .collect();

        write!(
            f,
            r#"{{"code":{code},"kind":"{kind}","message":"{primary}","context":[{ctx}],"hints":[{hints}],"chain_depth":{depth}}}"#,
            code    = code,
            kind    = kind,
            primary = primary,
            ctx     = ctx_json.join(","),
            hints   = hints.join(","),
            depth   = depth,
        )
    }
}

impl fmt::Display for ErrorReporter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.mode {
            ReportMode::Compact => self.fmt_compact(f),
            ReportMode::Full    => self.fmt_full(f),
            ReportMode::Json    => self.fmt_json(f),
        }
    }
}

// ─── convenience method on MohuError ─────────────────────────────────────────

impl MohuError {
    /// Returns a compact single-line reporter for this error.
    pub fn report(&self) -> ErrorReporter<'_> {
        ErrorReporter::compact(self)
    }

    /// Returns a full multi-line reporter for this error.
    pub fn report_full(&self) -> ErrorReporter<'_> {
        ErrorReporter::full(self)
    }

    /// Returns a JSON reporter for this error.
    pub fn report_json(&self) -> ErrorReporter<'_> {
        ErrorReporter::json(self)
    }
}

// ─── severity ─────────────────────────────────────────────────────────────────

/// The severity of an error — used by logging integrations to decide
/// whether to emit `warn!` or `error!`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Non-fatal — the operation failed but mohu is in a consistent state.
    Warning,
    /// The operation failed. The error should be propagated.
    Error,
    /// A critical invariant was violated. Should usually panic or abort.
    Fatal,
}

impl Severity {
    /// Returns a human-readable lowercase label for this severity level.
    pub fn label(self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::Error   => "error",
            Self::Fatal   => "fatal",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

impl MohuError {
    /// Returns the [`Severity`] of this error.
    ///
    /// `Internal` errors are `Fatal`. Everything else is `Error`.
    /// (There are currently no `Warning`-severity errors — that is
    /// reserved for a future lenient/strict mode API.)
    pub fn severity(&self) -> Severity {
        match ErrorChain::root(self) {
            MohuError::Internal(_) => Severity::Fatal,
            _                      => Severity::Error,
        }
    }
}
