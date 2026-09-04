use std::{convert::TryFrom, marker::PhantomData};

use mohu_buffer::buffer::Buffer;
use mohu_dtype::dtype::DType;
use mohu_dtype::scalar::Scalar;
use mohu_error::{MohuError, MohuResult};

/// Element types supported by [`NdArray`].
pub trait MohuElement: Scalar {}

impl<T: Scalar> MohuElement for T {}

/// Typed N-dimensional array backed by [`Buffer`].
pub struct NdArray<T: MohuElement> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T: MohuElement> NdArray<T> {
    /// Creates an array filled with zero values.
    pub fn zeros(shape: &[usize]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::zeros(T::DTYPE, shape)?,
            _marker: PhantomData,
        })
    }

    /// Creates an array filled with one values.
    pub fn ones(shape: &[usize]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::ones(T::DTYPE, shape)?,
            _marker: PhantomData,
        })
    }

    /// Creates a one-dimensional array by copying `data`.
    pub fn from_slice(data: &[T]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::from_slice(data)?,
            _marker: PhantomData,
        })
    }

    /// Creates an array by copying flat row-major data into `shape`.
    pub fn from_shape_slice(shape: &[usize], data: &[T]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::from_slice(data)?.reshape(shape)?,
            _marker: PhantomData,
        })
    }

    /// Returns the array shape.
    pub fn shape(&self) -> &[usize] {
        self.buffer.shape()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.buffer.ndim()
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the runtime dtype of the backing buffer.
    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }

    /// Copies the logical elements into a typed vector.
    pub fn to_vec(&self) -> MohuResult<Vec<T>> {
        self.buffer.to_vec::<T>()
    }

    /// Borrows the underlying buffer for read-only integration.
    pub fn as_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Consumes the array and returns its underlying buffer without cloning.
    pub fn into_buffer(self) -> Buffer {
        self.buffer
    }
}

impl<T: MohuElement> TryFrom<Buffer> for NdArray<T> {
    type Error = MohuError;

    fn try_from(buffer: Buffer) -> MohuResult<Self> {
        if buffer.dtype() != T::DTYPE {
            return Err(MohuError::DTypeMismatch {
                expected: T::DTYPE.to_string(),
                got: buffer.dtype().to_string(),
            });
        }
        Ok(Self {
            buffer,
            _marker: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::NdArray;

    #[test]
    fn constructors_store_expected_values() {
        let zeros = NdArray::<f64>::zeros(&[2]).unwrap();
        assert_eq!(zeros.buffer.to_vec::<f64>().unwrap(), vec![0.0, 0.0]);

        let ones = NdArray::<f32>::ones(&[2]).unwrap();
        assert_eq!(ones.buffer.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);

        let data = [3_i32, -2, 7];
        let copied = NdArray::<i32>::from_slice(&data).unwrap();
        assert_eq!(copied.buffer.to_vec::<i32>().unwrap(), data);

        let shaped = NdArray::<i32>::from_shape_slice(&[1, 3], &data).unwrap();
        assert_eq!(shaped.buffer.to_vec::<i32>().unwrap(), data);
    }
}
