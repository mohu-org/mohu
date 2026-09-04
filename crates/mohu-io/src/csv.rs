//! Typed, homogeneous CSV input for [`mohu_core::mohu_array::NdArray`].

use std::{fmt::Display, fs::File, path::Path, str::FromStr};

use mohu_core::{
    mohu_array::{MohuElement, NdArray},
    mohu_error::{MohuError, MohuResult},
};

/// Options controlling typed CSV input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsvReadOptions {
    /// Single-byte field delimiter.
    pub delimiter: u8,
    /// Whether the first record is a header and should be skipped.
    pub has_header: bool,
}

impl Default for CsvReadOptions {
    fn default() -> Self {
        Self {
            delimiter: b',',
            has_header: false,
        }
    }
}

/// Reads a rectangular, homogeneous CSV file into a typed 2-D array.
///
/// Dtype is explicit in `T`; v1 does not infer types, preserve header names,
/// represent missing values, or write CSV files. The first record is treated
/// as data unless `options.has_header` is set.
pub fn read_csv<T, P>(path: P, options: CsvReadOptions) -> MohuResult<NdArray<T>>
where
    T: MohuElement + FromStr,
    T::Err: Display,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    read_csv_reader(file, options)
}

fn read_csv_reader<T, R>(reader: R, options: CsvReadOptions) -> MohuResult<NdArray<T>>
where
    T: MohuElement + FromStr,
    T::Err: Display,
    R: std::io::Read,
{
    if options.delimiter == 0 {
        return Err(csv_error(0, 0, "NUL is not a valid CSV delimiter"));
    }

    let mut csv_reader = csv::ReaderBuilder::new()
        .delimiter(options.delimiter)
        .has_headers(false)
        .from_reader(reader);
    let mut records = csv_reader.records();
    let mut row = 0usize;
    let ncols = if options.has_header {
        let header = records
            .next()
            .ok_or_else(|| csv_error(0, 0, "CSV header is missing"))?
            .map_err(|error| csv_error(0, 0, error.to_string()))?;
        if header.is_empty() {
            return Err(csv_error(0, 0, "CSV header has no columns"));
        }
        row = 1;
        header.len()
    } else {
        let first = records
            .next()
            .ok_or_else(|| csv_error(0, 0, "CSV contains no records"))?
            .map_err(|error| csv_error(0, 0, error.to_string()))?;
        if first.is_empty() {
            return Err(csv_error(0, 0, "CSV record has no columns"));
        }
        let ncols = first.len();
        let mut values = parse_record::<T>(&first, row)?;
        row += 1;
        for record in records {
            let record = record.map_err(|error| csv_error(row, 0, error.to_string()))?;
            if record.len() != ncols {
                return Err(csv_error(
                    row,
                    0,
                    format!("expected {ncols} columns, got {}", record.len()),
                ));
            }
            values.extend(parse_record::<T>(&record, row)?);
            row += 1;
        }
        return NdArray::from_shape_slice(&[row, ncols], &values);
    };

    let mut values = Vec::new();
    for record in records {
        let record = record.map_err(|error| csv_error(row, 0, error.to_string()))?;
        if record.len() != ncols {
            return Err(csv_error(
                row,
                0,
                format!("expected {ncols} columns, got {}", record.len()),
            ));
        }
        values.extend(parse_record::<T>(&record, row)?);
        row += 1;
    }
    NdArray::from_shape_slice(&[row - 1, ncols], &values)
}

fn parse_record<T>(record: &csv::StringRecord, row: usize) -> MohuResult<Vec<T>>
where
    T: FromStr,
    T::Err: Display,
{
    record
        .iter()
        .enumerate()
        .map(|(col, field)| {
            field
                .parse::<T>()
                .map_err(|error| csv_error(row, col, error.to_string()))
        })
        .collect()
}

fn csv_error(row: usize, col: usize, detail: impl Into<String>) -> MohuError {
    MohuError::CsvParseError {
        row,
        col,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn reads_typed_rectangular_data() {
        let array =
            read_csv_reader::<i64, _>(Cursor::new("1,2\n-3,4\n"), CsvReadOptions::default())
                .unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert_eq!(array.dtype(), mohu_core::mohu_dtype::dtype::DType::I64);
    }

    #[test]
    fn parses_values_in_row_major_order() {
        let record = csv::StringRecord::from(vec!["1.5", "-2", "3"]);
        let values = parse_record::<f32>(&record, 0).unwrap();
        assert_eq!(values, vec![1.5, -2.0, 3.0]);
    }

    #[test]
    fn supports_header_and_custom_delimiter() {
        let array = read_csv_reader::<f32, _>(
            Cursor::new("x;y\n1.5;2.5\n"),
            CsvReadOptions {
                delimiter: b';',
                has_header: true,
            },
        )
        .unwrap();
        assert_eq!(array.shape(), &[1, 2]);
    }

    #[test]
    fn rejects_ragged_and_unparseable_records_with_context() {
        let ragged = read_csv_reader::<i64, _>(Cursor::new("1,2\n3\n"), CsvReadOptions::default())
            .err()
            .expect("expected CSV error");
        assert!(matches!(ragged, MohuError::CsvParseError { row: 1, .. }));
        let bad = read_csv_reader::<i64, _>(Cursor::new("1,nope\n"), CsvReadOptions::default())
            .err()
            .expect("expected CSV error");
        assert!(matches!(
            bad,
            MohuError::CsvParseError { row: 0, col: 1, .. }
        ));
    }

    #[test]
    fn handles_empty_and_header_only_files() {
        let empty = read_csv_reader::<i64, _>(Cursor::new(""), CsvReadOptions::default())
            .err()
            .expect("expected CSV error");
        assert!(matches!(empty, MohuError::CsvParseError { .. }));
        let header = read_csv_reader::<i64, _>(
            Cursor::new("a,b,c\n"),
            CsvReadOptions {
                has_header: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(header.shape(), &[0, 3]);
    }
}
