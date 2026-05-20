import pytest
import mohu

def test_shape_attribute():
    arr = mohu.zeros((2, 3))
    assert arr.shape == (2, 3)

def test_ndim_attribute():
    arr = mohu.zeros((2, 3, 4))
    assert arr.ndim == 3

def test_size_attribute():
    arr = mohu.zeros((2, 3))
    assert arr.size == 6

def test_reshape():
    arr = mohu.zeros((2, 3))
    arr_reshaped = arr.reshape((3, 2))
    assert arr_reshaped.shape == (3, 2)

def test_transpose():
    arr = mohu.zeros((2, 3))
    arr_transposed = arr.transpose()
    # Depending on implementation, transpose might flip all axes or just 2D
    # For a 2x3, it should become 3x2.
    assert arr_transposed.shape == (3, 2)
