import pytest
import mohu

def test_zeros():
    arr = mohu.zeros((2, 3))
    assert arr.shape == (2, 3)

def test_zeros_dtype():
    arr = mohu.zeros((4,), dtype="int32")
    assert str(arr.dtype) == "int32"

def test_ones():
    arr = mohu.ones((4,))
    assert arr.shape == (4,)

def test_full():
    arr = mohu.full((2, 2), 7.0)
    assert arr.shape == (2, 2)

def test_from_list_1d():
    arr = mohu.from_list([1, 2, 3])
    assert arr.shape == (3,)

def test_from_list_2d():
    arr = mohu.from_list([[1, 2], [3, 4]])
    assert arr.shape == (2, 2)
