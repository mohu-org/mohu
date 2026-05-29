//! CSV file I/O for mohu arrays.
//!
//! This module provides a small, typed CSV reader and writer with custom
//! delimiters, header handling, missing-value sentinels, and streaming parse
//! behavior suitable for large inputs.

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use thiserror::Error;

/// Errors produced while reading or writing CSV data.
#[derive(Debug, Error)]
pub enum CsvError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Failure reported by the `csv` crate.
    #[error("CSV parse error: {0}")]
    Parse(#[from] csv::Error),

    /// UTF-8 conversion failed while producing a `String` output.
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    /// No header or data rows were found.
    #[error("empty file: no records found")]
    EmptyFile,

    /// A data row had a different number of columns than expected.
    #[error("row {row} has {got} columns, expected {expected}")]
    ColumnMismatch {
        /// 1-based row index within the data section.
        row: usize,
        /// Expected number of columns.
        expected: usize,
        /// Actual number of columns.
        got: usize,
    },
}

/// Result type used by CSV reader and writer operations.
pub type CsvResult<T> = Result<T, CsvError>;

/// A single typed cell value inferred from a CSV field.
#[derive(Debug, Clone, PartialEq)]
pub enum CsvValue {
    /// Signed integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// String value.
    Str(String),
    /// Missing value placeholder.
    Missing,
}

impl CsvValue {
    /// Infer a typed value from a CSV field.
    fn infer(raw: &str, missing_values: &[&str]) -> Self {
        let trimmed = raw.trim();

        if trimmed.is_empty() || missing_values.contains(&trimmed) {
            return Self::Missing;
        }

        if let Ok(value) = trimmed.parse::<i64>() {
            return Self::Int(value);
        }

        if let Ok(value) = trimmed.parse::<f64>() {
            return Self::Float(value);
        }

        match trimmed.to_ascii_lowercase().as_str() {
            "true" | "yes" => Self::Bool(true),
            "false" | "no" => Self::Bool(false),
            _ => Self::Str(trimmed.to_owned()),
        }
    }

    /// Convert the value into a CSV-ready string.
    pub fn to_csv_string(&self) -> String {
        match self {
            Self::Int(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Str(value) => value.clone(),
            Self::Missing => String::new(),
        }
    }
}

/// Read configuration for [`CsvReader`].
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Field delimiter.
    pub delimiter: u8,
    /// Whether the first non-comment row should be treated as a header.
    pub has_header: bool,
    /// Values treated as missing.
    pub missing_values: Vec<String>,
    /// Optional comment prefix.
    pub comment: Option<u8>,
    /// Maximum number of data rows to read.
    pub max_rows: Option<usize>,
    /// Number of data rows to skip after the header.
    pub skip_rows: usize,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: true,
            missing_values: vec![
                String::new(),
                "NA".to_owned(),
                "N/A".to_owned(),
                "nan".to_owned(),
                "NaN".to_owned(),
                "null".to_owned(),
                "NULL".to_owned(),
            ],
            comment: None,
            max_rows: None,
            skip_rows: 0,
        }
    }
}

/// Parsed CSV content stored in row-major form.
#[derive(Debug, Clone)]
pub struct CsvTable {
    /// Column names. Empty when the input had no header.
    pub headers: Vec<String>,
    /// Row-major cells.
    pub data: Vec<Vec<CsvValue>>,
    /// Number of columns.
    pub ncols: usize,
}

impl CsvTable {
    /// Number of data rows.
    pub fn nrows(&self) -> usize {
        self.data.len()
    }

    /// Return a column by index.
    pub fn column(&self, idx: usize) -> Option<Vec<&CsvValue>> {
        if idx >= self.ncols {
            return None;
        }

        Some(self.data.iter().map(|row| &row[idx]).collect())
    }

    /// Return a column by header name.
    pub fn column_by_name(&self, name: &str) -> Option<Vec<&CsvValue>> {
        let idx = self.headers.iter().position(|header| header == name)?;
        self.column(idx)
    }
}

/// Reads a CSV file into a [`CsvTable`].
///
/// # Example
/// ```rust,no_run
/// use mohu_io::csv::{CsvReader, ReadOptions};
///
/// let table = CsvReader::new(ReadOptions::default())
///     .read_file("data.csv")
///     .unwrap();
///
/// assert!(table.nrows() > 0);
/// ```
pub struct CsvReader {
    opts: ReadOptions,
}

impl CsvReader {
    /// Create a reader with the given options.
    pub fn new(opts: ReadOptions) -> Self {
        Self { opts }
    }

    /// Read CSV content from a file path.
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> CsvResult<CsvTable> {
        let file = File::open(path)?;
        self.read_impl(BufReader::new(file))
    }

    /// Read CSV content from an in-memory string.
    pub fn read_str(&self, src: &str) -> CsvResult<CsvTable> {
        self.read_impl(src.as_bytes())
    }

    fn read_impl<R: Read>(&self, reader: R) -> CsvResult<CsvTable> {
        let missing_values: Vec<&str> = self
            .opts
            .missing_values
            .iter()
            .map(String::as_str)
            .collect();

        let mut csv_reader = csv::ReaderBuilder::new()
            .delimiter(self.opts.delimiter)
            .has_headers(false)
            .comment(self.opts.comment)
            .trim(csv::Trim::All)
            .flexible(true)
            .from_reader(reader);

        let mut records = csv_reader.records();

        let headers = if self.opts.has_header {
            let header_record = match records.next() {
                Some(result) => result?,
                None => return Err(CsvError::EmptyFile),
            };

            header_record
                .iter()
                .map(|value| value.trim().to_owned())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut data = Vec::new();
        let mut ncols = if headers.is_empty() {
            None
        } else {
            Some(headers.len())
        };

        for (record_index, result) in records.enumerate() {
            if record_index < self.opts.skip_rows {
                continue;
            }

            if let Some(limit) = self.opts.max_rows {
                if data.len() >= limit {
                    break;
                }
            }

            let record = result?;
            let cells = record
                .iter()
                .map(|field| CsvValue::infer(field, &missing_values))
                .collect::<Vec<_>>();

            match ncols {
                None => ncols = Some(cells.len()),
                Some(expected) if cells.len() != expected => {
                    return Err(CsvError::ColumnMismatch {
                        row: record_index + 1,
                        expected,
                        got: cells.len(),
                    });
                },
                Some(_) => {},
            }

            data.push(cells);
        }

        let ncols = ncols.unwrap_or(0);

        if data.is_empty() {
            return Err(CsvError::EmptyFile);
        }

        Ok(CsvTable {
            headers,
            data,
            ncols,
        })
    }
}

/// Write configuration for [`CsvWriter`].
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Field delimiter.
    pub delimiter: u8,
    /// Whether to write the header row.
    pub write_header: bool,
    /// Representation to use for missing values.
    pub missing_repr: String,
    /// Line terminator written after each row.
    pub line_terminator: String,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            write_header: true,
            missing_repr: String::new(),
            line_terminator: "\n".to_owned(),
        }
    }
}

/// Writes a [`CsvTable`] to a file or string.
///
/// # Example
/// ```rust,no_run
/// use mohu_io::csv::{CsvWriter, WriteOptions};
///
/// # let table = mohu_io::csv::CsvTable {
/// #     headers: vec!["a".to_owned()],
/// #     data: vec![vec![mohu_io::csv::CsvValue::Int(1)]],
/// #     ncols: 1,
/// # };
/// let csv = CsvWriter::new(WriteOptions::default()).write_str(&table).unwrap();
/// assert!(csv.contains('a'));
/// ```
pub struct CsvWriter {
    opts: WriteOptions,
}

impl CsvWriter {
    /// Create a writer with the given options.
    pub fn new(opts: WriteOptions) -> Self {
        Self { opts }
    }

    /// Write a [`CsvTable`] to a file.
    pub fn write_file<P: AsRef<Path>>(&self, table: &CsvTable, path: P) -> CsvResult<()> {
        let file = File::create(path)?;
        self.write_impl(table, BufWriter::new(file))
    }

    /// Write a [`CsvTable`] to a string.
    pub fn write_str(&self, table: &CsvTable) -> CsvResult<String> {
        let mut buffer = Vec::new();
        self.write_impl(table, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }

    fn write_impl<W: Write>(&self, table: &CsvTable, mut writer: W) -> CsvResult<()> {
        if self.opts.write_header && !table.headers.is_empty() {
            self.write_row(&mut writer, table.headers.iter().map(String::as_str))?;
        }

        for row in &table.data {
            let values = row
                .iter()
                .map(|value| {
                    if matches!(value, CsvValue::Missing) {
                        self.opts.missing_repr.as_str().to_owned()
                    } else {
                        value.to_csv_string()
                    }
                })
                .collect::<Vec<_>>();

            self.write_row(&mut writer, values.iter().map(String::as_str))?;
        }

        Ok(())
    }

    fn write_row<W, I, S>(&self, writer: &mut W, fields: I) -> CsvResult<()>
    where
        W: Write,
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut row_bytes = Vec::new();
        {
            let mut csv_writer = csv::WriterBuilder::new()
                .delimiter(self.opts.delimiter)
                .has_headers(false)
                .terminator(csv::Terminator::Any(b'\n'))
                .from_writer(&mut row_bytes);

            csv_writer.write_record(fields)?;
            csv_writer.flush()?;
        }

        if self.opts.line_terminator == "\n" {
            writer.write_all(&row_bytes)?;
            return Ok(());
        }

        if row_bytes.ends_with(b"\r\n") {
            row_bytes.truncate(row_bytes.len().saturating_sub(2));
        } else if row_bytes.ends_with(b"\n") {
            row_bytes.pop();
        }

        writer.write_all(&row_bytes)?;
        writer.write_all(self.opts.line_terminator.as_bytes())?;
        Ok(())
    }
}

/// Read a CSV file using default options.
pub fn read_csv<P: AsRef<Path>>(path: P) -> CsvResult<CsvTable> {
    CsvReader::new(ReadOptions::default()).read_file(path)
}

/// Write a [`CsvTable`] to a file using default options.
pub fn write_csv<P: AsRef<Path>>(table: &CsvTable, path: P) -> CsvResult<()> {
    CsvWriter::new(WriteOptions::default()).write_file(table, path)
}
