//! NumPy `.npy` / `.npz` serialization for [`Buffer`](mohu_buffer::Buffer).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use mohu_core::{
    mohu_buffer::Buffer,
    mohu_dtype::{ALL_DTYPES, DType},
    mohu_error::{MohuError, MohuResult},
};
use num_complex::{Complex32, Complex64};
use zip::ZipWriter;
use zip::read::ZipArchive;
use zip::write::SimpleFileOptions;

const NPY_MAGIC: &[u8] = b"\x93NUMPY";
const NPY_VERSION: [u8; 2] = [1, 0];

/// Serializes a C-contiguous buffer to the NumPy `.npy` file format.
pub fn save_npy(path: impl AsRef<Path>, buffer: &Buffer) -> MohuResult<()> {
    std::fs::write(path, write_npy_bytes(buffer)?).map_err(MohuError::Io)
}

/// Loads a single-array `.npy` file into a C-contiguous [`Buffer`].
pub fn load_npy(path: impl AsRef<Path>) -> MohuResult<Buffer> {
    let bytes = std::fs::read(path).map_err(MohuError::Io)?;
    load_npy_bytes(&bytes)
}

/// Writes multiple named arrays into a `.npz` ZIP archive of `.npy` members.
pub fn save_npz(path: impl AsRef<Path>, arrays: &[(&str, &Buffer)]) -> MohuResult<()> {
    let file = File::create(path.as_ref()).map_err(MohuError::Io)?;
    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

    for (name, buffer) in arrays {
        let entry = format!("{name}.npy");
        zip.start_file(entry, options)
            .map_err(|e| MohuError::Io(e.into()))?;
        zip.write_all(&write_npy_bytes(buffer)?)
            .map_err(|e| MohuError::Io(e.into()))?;
    }

    zip.finish().map_err(|e| MohuError::Io(e.into()))?;
    Ok(())
}

/// Loads all `.npy` members from a `.npz` archive.
pub fn load_npz(path: impl AsRef<Path>) -> MohuResult<HashMap<String, Buffer>> {
    let file = File::open(path.as_ref()).map_err(MohuError::Io)?;
    let mut archive = ZipArchive::new(file).map_err(|e| MohuError::Io(e.into()))?;
    let mut out = HashMap::new();

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|e| MohuError::Io(e.into()))?;
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|e| MohuError::Io(e.into()))?;
        let key = entry
            .name()
            .rsplit('/')
            .next()
            .unwrap_or(entry.name())
            .strip_suffix(".npy")
            .unwrap_or(entry.name())
            .to_string();
        out.insert(key, load_npy_bytes(&bytes)?);
    }

    Ok(out)
}

fn write_npy_bytes(buffer: &Buffer) -> MohuResult<Vec<u8>> {
    if !buffer.is_c_contiguous() {
        return Err(MohuError::NonContiguous);
    }

    let header = build_header(buffer.dtype(), buffer.shape())?;
    let mut out = Vec::with_capacity(NPY_MAGIC.len() + 2 + 2 + header.len() + buffer.nbytes());
    out.extend_from_slice(NPY_MAGIC);
    out.extend_from_slice(&NPY_VERSION);
    let header_len = u16::try_from(header.len()).map_err(|_| MohuError::NpyHeaderError {
        detail: "npy header exceeds 65535 bytes".into(),
    })?;
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header);
    copy_buffer_bytes(buffer, &mut out)?;
    Ok(out)
}

fn build_header(dtype: DType, shape: &[usize]) -> MohuResult<Vec<u8>> {
    let descr = dtype.array_interface_typestr();
    let shape_repr = format_shape(shape);
    let mut dict =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}");
    dict.push('\n');

    let prefix_len = NPY_MAGIC.len() + NPY_VERSION.len() + 2;
    let pad = (64 - (prefix_len + dict.len()) % 64) % 64;
    dict.extend(std::iter::repeat(' ').take(pad));
    Ok(dict.into_bytes())
}

fn format_shape(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [dim] => format!("({dim},)"),
        dims => format!(
            "({})",
            dims.iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn copy_buffer_bytes(buffer: &Buffer, out: &mut Vec<u8>) -> MohuResult<()> {
    let nbytes = buffer.nbytes();
    // SAFETY: `buffer` is C-contiguous; `out` has reserved capacity.
    unsafe {
        let dst = out.as_mut_ptr().add(out.len());
        std::ptr::copy_nonoverlapping(buffer.as_ptr(), dst, nbytes);
        out.set_len(out.len() + nbytes);
    }
    Ok(())
}

fn load_npy_bytes(bytes: &[u8]) -> MohuResult<Buffer> {
    if bytes.len() < 10 || &bytes[..6] != NPY_MAGIC {
        return Err(MohuError::InvalidMagic {
            format: "npy",
            expected: NPY_MAGIC.to_vec(),
            got: bytes[..bytes.len().min(6)].to_vec(),
        });
    }

    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_end = 10 + header_len;
    if bytes.len() < header_end {
        return Err(MohuError::UnexpectedEof {
            format: "npy",
            offset: bytes.len() as u64,
        });
    }

    let header =
        std::str::from_utf8(&bytes[10..header_end]).map_err(|e| MohuError::NpyHeaderError {
            detail: e.to_string(),
        })?;

    let (descr, shape) = parse_header(header)?;
    let dtype = dtype_from_descr(&descr)?;
    let data = &bytes[header_end..];
    buffer_from_le_bytes(dtype, &shape, data)
}

fn parse_header(header: &str) -> MohuResult<(String, Vec<usize>)> {
    let descr = extract_quoted(header, "'descr':").ok_or_else(|| MohuError::NpyHeaderError {
        detail: "missing descr in npy header".into(),
    })?;
    let shape = parse_shape(header).ok_or_else(|| MohuError::NpyHeaderError {
        detail: "missing shape in npy header".into(),
    })?;
    Ok((descr, shape))
}

fn extract_quoted(header: &str, key: &str) -> Option<String> {
    let start = header.find(key)? + key.len();
    let rest = header[start..].trim_start();
    let quote = rest.chars().next()?;
    let end = rest[1..].find(quote)? + 1;
    Some(rest[1..end].to_string())
}

fn parse_shape(header: &str) -> Option<Vec<usize>> {
    let start = header.find("'shape':")? + "'shape':".len();
    let rest = header[start..].trim_start();
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    let inner = rest[open + 1..close].trim();
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        dims.push(part.parse::<usize>().ok()?);
    }
    Some(dims)
}

fn dtype_from_descr(descr: &str) -> MohuResult<DType> {
    ALL_DTYPES
        .iter()
        .copied()
        .find(|dtype| dtype.array_interface_typestr() == descr)
        .ok_or_else(|| MohuError::UnknownDType(descr.into()))
}

fn buffer_from_le_bytes(dtype: DType, shape: &[usize], data: &[u8]) -> MohuResult<Buffer> {
    let count: usize = shape.iter().product();
    let expected = count
        .checked_mul(dtype.itemsize())
        .ok_or(MohuError::ShapeOverflow { max: usize::MAX })?;
    if data.len() != expected {
        return Err(MohuError::CorruptData {
            format: "npy",
            detail: format!("expected {expected} data bytes, got {}", data.len()),
        });
    }

    macro_rules! load_le_slice {
        ($ty:ty) => {{
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(std::mem::size_of::<$ty>()) {
                values.push(<$ty>::from_le_bytes(chunk.try_into().unwrap()));
            }
            Buffer::from_slice(&values)?.reshape(shape)
        }};
    }

    match dtype {
        DType::Bool => {
            let values: Vec<bool> = data.iter().map(|byte| *byte != 0).collect();
            Buffer::from_slice(&values)?.reshape(shape)
        },
        DType::I8 => load_le_slice!(i8),
        DType::I16 => load_le_slice!(i16),
        DType::I32 => load_le_slice!(i32),
        DType::I64 => load_le_slice!(i64),
        DType::U8 => load_le_slice!(u8),
        DType::U16 => load_le_slice!(u16),
        DType::U32 => load_le_slice!(u32),
        DType::U64 => load_le_slice!(u64),
        DType::F32 => load_le_slice!(f32),
        DType::F64 => load_le_slice!(f64),
        DType::C64 => {
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(8) {
                let re = f32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let im = f32::from_le_bytes(chunk[4..8].try_into().unwrap());
                values.push(Complex32::new(re, im));
            }
            Buffer::from_slice(&values)?.reshape(shape)
        },
        DType::C128 => {
            let mut values = Vec::with_capacity(count);
            for chunk in data.chunks_exact(16) {
                let re = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
                let im = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
                values.push(Complex64::new(re, im));
            }
            Buffer::from_slice(&values)?.reshape(shape)
        },
        other => Err(MohuError::UnsupportedDType {
            op: "npy load",
            dtype: other.to_string(),
        }),
    }
}
