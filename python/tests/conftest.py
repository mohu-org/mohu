import pytest
import mohu

@pytest.fixture
def sample_int_array():
    return mohu.from_list([1, 2, 3, 4, 5])

@pytest.fixture
def sample_float_array():
    return mohu.from_list([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def sample_2d_array():
    return mohu.from_list([[1, 2], [3, 4]])
