use std::marker::PhantomData;

use mohu_buffer::buffer::Buffer;
use mohu_dtype::dtype::DType;
use mohu_dtype::scalar::Scalar;
use mohu_error::MohuResult;

/// Trait for types that can be elements of an `NdArray`.
///
/// All [`Scalar`] types automatically implement `MohuElement`.
/// This trait serves as a bound for the element type parameter
/// of `NdArray`, ensuring only valid numeric types are used.
pub trait MohuElement: Scalar {}

impl<T: Scalar> MohuElement for T {}

/// An N-dimensional array with typed elements.
///
/// `NdArray<T>` wraps a [`Buffer`] with a compile-time element type,
/// providing a type-safe API for construction, shape queries, and
/// dtype identification.
///
/// # Type parameters
///
/// * `T` — The element type. Must implement [`MohuElement`].
///
/// # Construction
///
/// ```rust
/// use mohu_array::NdArray;
///
/// let a = NdArray::<f64>::zeros(&[3, 4]).unwrap();
/// assert_eq!(a.shape(), &[3, 4]);
///
/// let b = NdArray::<f32>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
/// assert_eq!(b.len(), 3);
/// ```
pub struct NdArray<T: MohuElement> {
    buffer:  Buffer,
    _marker: PhantomData<T>,
}

impl<T: MohuElement> NdArray<T> {
    /// Creates a new `NdArray` filled with zeros.
    ///
    /// The shape is specified as a slice of dimension sizes.
    /// Returns an error if the shape would overflow the address space.
    pub fn zeros(shape: &[usize]) -> MohuResult<Self> {
        let buffer = Buffer::zeros(T::DTYPE, shape)?;
        Ok(Self { buffer, _marker: PhantomData })
    }

    /// Creates a new `NdArray` filled with ones.
    ///
    /// The shape is specified as a slice of dimension sizes.
    /// Returns an error if the shape would overflow the address space.
    pub fn ones(shape: &[usize]) -> MohuResult<Self> {
        let buffer = Buffer::ones(T::DTYPE, shape)?;
        Ok(Self { buffer, _marker: PhantomData })
    }

    /// Creates a 1-D `NdArray` by copying elements from a slice.
    ///
    /// The resulting array is always 1-dimensional with length equal to
    /// `data.len()`. Returns an error if the allocation fails.
    pub fn from_slice(data: &[T]) -> MohuResult<Self> {
        let buffer = Buffer::from_slice(data)?;
        Ok(Self { buffer, _marker: PhantomData })
    }

    /// Returns the shape of the array as a slice of dimension sizes.
    ///
    /// The returned slice has length equal to `ndim()`.
    pub fn shape(&self) -> &[usize] {
        self.buffer.shape()
    }

    /// Returns the number of dimensions (axes) of the array.
    ///
    /// A scalar (0-dimensional) array has `ndim() == 0`.
    pub fn ndim(&self) -> usize {
        self.buffer.ndim()
    }

    /// Returns the total number of elements in the array.
    ///
    /// This is the product of all dimension sizes.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the array contains zero elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the data type of the array elements.
    ///
    /// This is always `T::DTYPE` at runtime, but queried from the
    /// underlying buffer for consistency with the dynamic-type API.
    pub fn dtype(&self) -> DType {
        T::DTYPE
    }
}

impl<T: MohuElement> Clone for NdArray<T> {
    fn clone(&self) -> Self {
        Self {
            buffer:  self.buffer.share(),
            _marker: PhantomData,
        }
    }
}

impl<T: MohuElement> std::fmt::Debug for NdArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NdArray")
            .field("shape", &self.shape())
            .field("dtype", &self.dtype())
            .field("len", &self.len())
            .finish()
    }
}
