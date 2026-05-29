use mohu_io::csv::{CsvError, CsvReader, CsvValue, CsvWriter, ReadOptions, WriteOptions};

const SAMPLE_CSV: &str = "\
name,age,score,active
Alice,30,9.5,true
Bob,25,8.1,false
Charlie,,7.7,true
";

const TAB_CSV: &str = "x\ty\tz\n1\t2\t3\n4\t5\t6\n";

#[test]
fn test_read_basic_headers() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    assert_eq!(table.headers, vec!["name", "age", "score", "active"]);
    assert_eq!(table.nrows(), 3);
    assert_eq!(table.ncols, 4);
}

#[test]
fn test_type_inference_int() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    assert_eq!(table.data[0][1], CsvValue::Int(30));
    assert_eq!(table.data[1][1], CsvValue::Int(25));
}

#[test]
fn test_type_inference_float() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    assert_eq!(table.data[0][2], CsvValue::Float(9.5));
}

#[test]
fn test_type_inference_bool() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    assert_eq!(table.data[0][3], CsvValue::Bool(true));
    assert_eq!(table.data[1][3], CsvValue::Bool(false));
}

#[test]
fn test_missing_value_detected() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    assert_eq!(table.data[2][1], CsvValue::Missing);
}

#[test]
fn test_custom_missing_sentinel() {
    let csv = "a,b\n1,N/A\n2,3\n";
    let opts = ReadOptions {
        missing_values: vec!["N/A".to_owned()],
        ..Default::default()
    };

    let table = CsvReader::new(opts).read_str(csv).unwrap();
    assert_eq!(table.data[0][1], CsvValue::Missing);
    assert_eq!(table.data[1][1], CsvValue::Int(3));
}

#[test]
fn test_tab_delimiter() {
    let opts = ReadOptions {
        delimiter: b'\t',
        ..Default::default()
    };

    let table = CsvReader::new(opts).read_str(TAB_CSV).unwrap();
    assert_eq!(table.headers, vec!["x", "y", "z"]);
    assert_eq!(table.data[0][0], CsvValue::Int(1));
}

#[test]
fn test_no_header() {
    let csv = "1,2,3\n4,5,6\n";
    let opts = ReadOptions {
        has_header: false,
        ..Default::default()
    };

    let table = CsvReader::new(opts).read_str(csv).unwrap();
    assert!(table.headers.is_empty());
    assert_eq!(table.nrows(), 2);
}

#[test]
fn test_max_rows() {
    let opts = ReadOptions {
        max_rows: Some(1),
        ..Default::default()
    };

    let table = CsvReader::new(opts).read_str(SAMPLE_CSV).unwrap();
    assert_eq!(table.nrows(), 1);
}

#[test]
fn test_skip_rows() {
    let opts = ReadOptions {
        skip_rows: 1,
        ..Default::default()
    };

    let table = CsvReader::new(opts).read_str(SAMPLE_CSV).unwrap();
    assert_eq!(table.data[0][0], CsvValue::Str("Bob".to_owned()));
}

#[test]
fn test_column_by_name() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    let names: Vec<_> = table.column_by_name("name").unwrap();
    assert_eq!(names[0], &CsvValue::Str("Alice".to_owned()));
}

#[test]
fn test_empty_file_error() {
    let result = CsvReader::new(ReadOptions::default()).read_str("name,age\n");
    assert!(matches!(result, Err(CsvError::EmptyFile)));
}

#[test]
fn test_round_trip() {
    let original = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    let written = CsvWriter::new(WriteOptions::default())
        .write_str(&original)
        .unwrap();

    let recovered = CsvReader::new(ReadOptions::default())
        .read_str(&written)
        .unwrap();

    assert_eq!(original.headers, recovered.headers);
    assert_eq!(original.nrows(), recovered.nrows());
    assert_eq!(original.data[0][0], recovered.data[0][0]);
}

#[test]
fn test_write_tab_delimiter() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    let opts = WriteOptions {
        delimiter: b'\t',
        ..Default::default()
    };

    let out = CsvWriter::new(opts).write_str(&table).unwrap();
    assert!(out.contains('\t'));
    assert!(!out.contains(','));
}

#[test]
fn test_write_missing_repr() {
    let csv = "a,b\n1,\n2,3\n";
    let table = CsvReader::new(ReadOptions::default())
        .read_str(csv)
        .unwrap();

    let opts = WriteOptions {
        missing_repr: "NA".to_owned(),
        ..Default::default()
    };

    let out = CsvWriter::new(opts).write_str(&table).unwrap();
    assert!(out.contains("NA"));
}

#[test]
fn test_write_no_header() {
    let table = CsvReader::new(ReadOptions::default())
        .read_str(SAMPLE_CSV)
        .unwrap();

    let opts = WriteOptions {
        write_header: false,
        ..Default::default()
    };

    let out = CsvWriter::new(opts).write_str(&table).unwrap();
    assert!(!out.starts_with("name"));
}