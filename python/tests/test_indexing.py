import pytest
import mohu

def test_single_element_get(sample_int_array):
    val = sample_int_array[0]
    # We just ensure it runs without error, maybe check it's not None
    assert val is not None

def test_single_element_set(sample_int_array):
    sample_int_array[0] = 10
    # verify
    assert sample_int_array[0] == 10

def test_slicing(sample_int_array):
    sub_arr = sample_int_array[1:3]
    assert sub_arr.shape == (2,)

def test_negative_indices(sample_int_array):
    val = sample_int_array[-1]
    assert val is not None

def test_negative_indices_set(sample_int_array):
    sample_int_array[-1] = 99
    assert sample_int_array[-1] == 99

def test_2d_indexing(sample_2d_array):
    val = sample_2d_array[1, 1]
    assert val is not None
