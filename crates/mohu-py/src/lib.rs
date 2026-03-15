use pyo3::prelude::*;

mod array;
mod ops;
mod linalg;
mod stats;
mod io;

#[pymodule]
fn mohu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    array::register(m)?;
    ops::register(m)?;
    linalg::register(m)?;
    stats::register(m)?;
    io::register(m)?;
    Ok(())
}
