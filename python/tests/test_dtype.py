import pytest
import mohu

def test_dtype_attribute(sample_int_array):
    assert hasattr(sample_int_array, "dtype")

def test_dtype_parsing_zeros():
    arr = mohu.zeros((2,), dtype="int32")
    assert str(arr.dtype) == "int32"

def test_astype(sample_int_array):
    arr_float = sample_int_array.astype("float32")
    assert str(arr_float.dtype) == "float32"

def test_dtype_mismatch_error_set():
    arr = mohu.zeros((2,), dtype="int32")
    with pytest.raises(TypeError):
        arr[0] = "string_value"

def test_dtype_mismatch_error_full():
    with pytest.raises(TypeError):
        mohu.full((2,), "invalid_fill", dtype="int32")
