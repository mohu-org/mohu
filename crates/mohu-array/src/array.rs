use std::marker::PhantomData;

use mohu_buffer::Buffer;
use mohu_dtype::{DType, Scalar};
use mohu_error::MohuResult;

pub trait MohuElement: Scalar {}

impl<T: Scalar> MohuElement for T {}

pub struct NdArray<T: MohuElement> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T: MohuElement> NdArray<T> {
    pub fn zeros(shape: &[usize]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::zeros(T::DTYPE, shape)?,
            _marker: PhantomData,
        })
    }

    pub fn ones(shape: &[usize]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::ones(T::DTYPE, shape)?,
            _marker: PhantomData,
        })
    }

    pub fn from_slice(data: &[T]) -> MohuResult<Self> {
        Ok(Self {
            buffer: Buffer::from_slice(data)?,
            _marker: PhantomData,
        })
    }

    pub fn shape(&self) -> &[usize] {
        self.buffer.shape()
    }

    pub fn ndim(&self) -> usize {
        self.buffer.ndim()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn dtype(&self) -> DType {
        self.buffer.dtype()
    }
}

#[cfg(test)]
mod tests {
    use super::NdArray;

    #[test]
    fn zeros_shapes_and_values() {
        let array = NdArray::<f64>::zeros(&[3, 4]).unwrap();
        assert_eq!(array.shape(), &[3, 4]);
        assert_eq!(array.ndim(), 2);
        assert_eq!(array.len(), 12);
        assert!(!array.is_empty());
        assert_eq!(array.dtype(), mohu_dtype::DType::F64);
        let values = array.buffer.as_slice::<f64>().unwrap();
        assert!(values.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn ones_and_from_slice() {
        let ones = NdArray::<f32>::ones(&[2, 3]).unwrap();
        let ones_values = ones.buffer.as_slice::<f32>().unwrap();
        assert!(ones_values.iter().all(|&value| value == 1.0));

        let array = NdArray::<f32>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(array.shape(), &[3]);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.len(), 3);
        assert!(!array.is_empty());
        assert_eq!(array.dtype(), mohu_dtype::DType::F32);
        assert_eq!(array.buffer.as_slice::<f32>().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
