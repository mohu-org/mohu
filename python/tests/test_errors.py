import pytest
import mohu

def test_ragged_list_creation():
    with pytest.raises(ValueError):
        mohu.from_list([[1, 2], [3]])

def test_shape_mismatch_reshape():
    arr = mohu.zeros((4,))
    with pytest.raises(ValueError):
        arr.reshape((3,))

def test_shape_mismatch_set():
    arr = mohu.zeros((4,))
    with pytest.raises(ValueError):
        arr[0:2] = mohu.from_list([1, 2, 3])

def test_invalid_negative_shape():
    with pytest.raises(ValueError):
        mohu.zeros((-1, 2))

def test_shape_mismatch_assign():
    arr = mohu.zeros((2, 2))
    with pytest.raises(ValueError):
        arr[0] = mohu.zeros((3,))
