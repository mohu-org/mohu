//! Arrow IPC serialization and deserialization for mohu buffers.
//!
//! Exposes four functions:
//!
//! | Function            | Description                                      |
//! |---------------------|--------------------------------------------------|
//! | [`read_ipc_file`]   | Read an Arrow IPC file into a `Vec<Buffer>`      |
//! | [`read_ipc_stream`] | Read an Arrow IPC stream into a `Vec<Buffer>`    |
//! | [`write_ipc_file`]  | Write `Buffer` slices to an Arrow IPC file       |
//! | [`write_ipc_stream`]| Write `Buffer` slices to an Arrow IPC stream     |
//!
//! # Arrow IPC formats
//!
//! - **File format**: random-access, seekable, has a footer. Suited for
//!   on-disk storage.
//! - **Streaming format**: sequential, no footer. Required for inter-process
//!   streaming (Polars pipe, DuckDB query results, Arrow Flight protocol).

use std::io::{Read, Seek, Write};
use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array, UInt8Array};
use arrow::datatypes::{DataType as ArrowDataType, Field, Schema};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, StreamWriter};
use arrow::record_batch::RecordBatch;

use mohu_core::mohu_buffer::Buffer;
use mohu_core::mohu_dtype::DType;
use mohu_core::mohu_error::{MohuError, MohuResult};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Maps a mohu `DType` to an Arrow `DataType`.
fn dtype_to_arrow(dtype: DType) -> MohuResult<ArrowDataType> {
    match dtype {
        DType::I8   => Ok(ArrowDataType::Int8),
        DType::I16  => Ok(ArrowDataType::Int16),
        DType::I32  => Ok(ArrowDataType::Int32),
        DType::I64  => Ok(ArrowDataType::Int64),
        DType::U8   => Ok(ArrowDataType::UInt8),
        DType::U16  => Ok(ArrowDataType::UInt16),
        DType::U32  => Ok(ArrowDataType::UInt32),
        DType::U64  => Ok(ArrowDataType::UInt64),
        DType::F32  => Ok(ArrowDataType::Float32),
        DType::F64  => Ok(ArrowDataType::Float64),
        other => Err(MohuError::UnsupportedDType {
            op: "arrow IPC",
            dtype: format!("{other:?}"),
        }),
    }
}

/// Maps an Arrow `DataType` to a mohu `DType`.
/// Maps an Arrow `DataType` to a mohu `DType`.


/// Converts a 1-D `Buffer` to an Arrow `ArrayRef`.
fn buffer_to_array(buf: &Buffer) -> MohuResult<ArrayRef> {
    match buf.dtype() {
        DType::I8  => {
            let s = buf.as_slice::<i8>()?;
            Ok(Arc::new(Int8Array::from(s.to_vec())))
        }
        DType::I16 => {
            let s = buf.as_slice::<i16>()?;
            Ok(Arc::new(Int16Array::from(s.to_vec())))
        }
        DType::I32 => {
            let s = buf.as_slice::<i32>()?;
            Ok(Arc::new(Int32Array::from(s.to_vec())))
        }
        DType::I64 => {
            let s = buf.as_slice::<i64>()?;
            Ok(Arc::new(Int64Array::from(s.to_vec())))
        }
        DType::U8  => {
            let s = buf.as_slice::<u8>()?;
            Ok(Arc::new(UInt8Array::from(s.to_vec())))
        }
        DType::U16 => {
            let s = buf.as_slice::<u16>()?;
            Ok(Arc::new(UInt16Array::from(s.to_vec())))
        }
        DType::U32 => {
            let s = buf.as_slice::<u32>()?;
            Ok(Arc::new(UInt32Array::from(s.to_vec())))
        }
        DType::U64 => {
            let s = buf.as_slice::<u64>()?;
            Ok(Arc::new(UInt64Array::from(s.to_vec())))
        }
        DType::F32 => {
            let s = buf.as_slice::<f32>()?;
            Ok(Arc::new(Float32Array::from(s.to_vec())))
        }
        DType::F64 => {
            let s = buf.as_slice::<f64>()?;
            Ok(Arc::new(Float64Array::from(s.to_vec())))
        }
        other => Err(MohuError::UnsupportedDType {
            op: "arrow IPC",
            dtype: format!("{other:?}"),
        }),
    }
}

/// Converts an Arrow `ArrayRef` to a 1-D `Buffer`.
fn array_to_buffer(array: &ArrayRef) -> MohuResult<Buffer> {
    match array.data_type() {
        ArrowDataType::Int8 => {
            let a = array.as_any().downcast_ref::<Int8Array>()
                .ok_or_else(|| MohuError::bug("downcast Int8Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::Int16 => {
            let a = array.as_any().downcast_ref::<Int16Array>()
                .ok_or_else(|| MohuError::bug("downcast Int16Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::Int32 => {
            let a = array.as_any().downcast_ref::<Int32Array>()
                .ok_or_else(|| MohuError::bug("downcast Int32Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::Int64 => {
            let a = array.as_any().downcast_ref::<Int64Array>()
                .ok_or_else(|| MohuError::bug("downcast Int64Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::UInt8 => {
            let a = array.as_any().downcast_ref::<UInt8Array>()
                .ok_or_else(|| MohuError::bug("downcast UInt8Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::UInt16 => {
            let a = array.as_any().downcast_ref::<UInt16Array>()
                .ok_or_else(|| MohuError::bug("downcast UInt16Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::UInt32 => {
            let a = array.as_any().downcast_ref::<UInt32Array>()
                .ok_or_else(|| MohuError::bug("downcast UInt32Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::UInt64 => {
            let a = array.as_any().downcast_ref::<UInt64Array>()
                .ok_or_else(|| MohuError::bug("downcast UInt64Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::Float32 => {
            let a = array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| MohuError::bug("downcast Float32Array failed"))?;
            Buffer::from_slice(a.values())
        }
        ArrowDataType::Float64 => {
            let a = array.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| MohuError::bug("downcast Float64Array failed"))?;
            Buffer::from_slice(a.values())
        }
        other => Err(MohuError::UnsupportedDType {
            op: "arrow IPC",
            dtype: format!("{other:?}"),
        }),
    }
}

/// Builds a `RecordBatch` from a slice of `Buffer`s.
fn buffers_to_record_batch(buffers: &[Buffer]) -> MohuResult<RecordBatch> {
    let mut fields = Vec::with_capacity(buffers.len());
    let mut arrays = Vec::with_capacity(buffers.len());

    for (i, buf) in buffers.iter().enumerate() {
        let arrow_dtype = dtype_to_arrow(buf.dtype())?;
        let field = Field::new(format!("col_{i}"), arrow_dtype, false);
        fields.push(field);
        arrays.push(buffer_to_array(buf)?);
    }

    let schema = Arc::new(Schema::new(fields));
    RecordBatch::try_new(schema, arrays)
        .map_err(|e| MohuError::ArrowSchema(e.to_string()))
}

/// Extracts `Buffer`s from a `RecordBatch`.
fn record_batch_to_buffers(batch: &RecordBatch) -> MohuResult<Vec<Buffer>> {
    batch
        .columns()
        .iter()
        .map(array_to_buffer)
        .collect()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Reads an Arrow IPC **file** from `reader` and returns the columns of
/// every record batch as a flat `Vec<Buffer>`.
///
/// The Arrow IPC file format is random-access and seekable, making it
/// suitable for on-disk storage.
///
/// # Errors
///
/// Returns an error if the IPC data is malformed or contains unsupported
/// Arrow data types.
pub fn read_ipc_file<R: Read + Seek>(reader: R) -> MohuResult<Vec<Buffer>> {
    let file_reader = FileReader::try_new(reader, None)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    let mut buffers = Vec::new();
    for batch in file_reader {
        let batch = batch.map_err(|e| MohuError::ArrowIpc(e.to_string()))?;
        buffers.extend(record_batch_to_buffers(&batch)?);
    }
    Ok(buffers)
}

/// Reads an Arrow IPC **stream** from `reader` and returns the columns of
/// every record batch as a flat `Vec<Buffer>`.
///
/// The Arrow IPC streaming format is sequential with no footer, making it
/// suitable for inter-process streaming (Polars pipe, DuckDB query results,
/// Arrow Flight protocol).
///
/// # Errors
///
/// Returns an error if the IPC data is malformed or contains unsupported
/// Arrow data types.
pub fn read_ipc_stream<R: Read>(reader: R) -> MohuResult<Vec<Buffer>> {
    let stream_reader = StreamReader::try_new(reader, None)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    let mut buffers = Vec::new();
    for batch in stream_reader {
        let batch = batch.map_err(|e| MohuError::ArrowIpc(e.to_string()))?;
        buffers.extend(record_batch_to_buffers(&batch)?);
    }
    Ok(buffers)
}

/// Writes `buffers` to `writer` in Arrow IPC **file** format.
///
/// All buffers are written as a single record batch where each buffer
/// becomes one column (`col_0`, `col_1`, …).
///
/// # Errors
///
/// Returns an error if any buffer has an unsupported dtype or if writing
/// fails.
pub fn write_ipc_file<W: Write>(writer: W, buffers: &[Buffer]) -> MohuResult<()> {
    if buffers.is_empty() {
        return Ok(());
    }

    let batch = buffers_to_record_batch(buffers)?;
    let schema = batch.schema();

    let mut file_writer = FileWriter::try_new(writer, &schema)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    file_writer
        .write(&batch)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    file_writer
        .finish()
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    Ok(())
}

/// Writes `buffers` to `writer` in Arrow IPC **stream** format.
///
/// All buffers are written as a single record batch where each buffer
/// becomes one column (`col_0`, `col_1`, …).
///
/// # Errors
///
/// Returns an error if any buffer has an unsupported dtype or if writing
/// fails.
pub fn write_ipc_stream<W: Write>(writer: W, buffers: &[Buffer]) -> MohuResult<()> {
    if buffers.is_empty() {
        return Ok(());
    }

    let batch = buffers_to_record_batch(buffers)?;
    let schema = batch.schema();

    let mut stream_writer = StreamWriter::try_new(writer, &schema)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    stream_writer
        .write(&batch)
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    stream_writer
        .finish()
        .map_err(|e| MohuError::ArrowIpc(e.to_string()))?;

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn round_trip_ipc_file_f32() {
        let buf = Buffer::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let mut bytes = Vec::new();
        write_ipc_file(&mut bytes, &[buf.clone()]).unwrap();

        let result = read_ipc_file(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice::<f32>().unwrap(), &[1.0_f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn round_trip_ipc_stream_f64() {
        let buf = Buffer::from_slice(&[10.0_f64, 20.0, 30.0]).unwrap();
        let mut bytes = Vec::new();
        write_ipc_stream(&mut bytes, &[buf.clone()]).unwrap();

        let result = read_ipc_stream(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice::<f64>().unwrap(), &[10.0_f64, 20.0, 30.0]);
    }

    #[test]
    fn round_trip_ipc_file_i32() {
        let buf = Buffer::from_slice(&[1_i32, 2, 3, 4, 5]).unwrap();
        let mut bytes = Vec::new();
        write_ipc_file(&mut bytes, &[buf.clone()]).unwrap();

        let result = read_ipc_file(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].as_slice::<i32>().unwrap(), &[1_i32, 2, 3, 4, 5]);
    }

    #[test]
    fn round_trip_ipc_stream_multiple_buffers() {
        let buf1 = Buffer::from_slice(&[1_i32, 2, 3]).unwrap();
        let buf2 = Buffer::from_slice(&[4.0_f64, 5.0, 6.0]).unwrap();
        let mut bytes = Vec::new();
        write_ipc_stream(&mut bytes, &[buf1, buf2]).unwrap();

        let result = read_ipc_stream(Cursor::new(bytes)).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].as_slice::<i32>().unwrap(), &[1_i32, 2, 3]);
        assert_eq!(result[1].as_slice::<f64>().unwrap(), &[4.0_f64, 5.0, 6.0]);
    }

    #[test]
    fn write_empty_buffers_is_ok() {
        let mut bytes = Vec::new();
        assert!(write_ipc_file(&mut bytes, &[]).is_ok());
        assert!(write_ipc_stream(&mut bytes, &[]).is_ok());
    }
}
