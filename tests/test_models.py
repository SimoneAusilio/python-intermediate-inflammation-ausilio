"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

import pytest
from inflammation.models import daily_mean, daily_max, daily_min, patient_normalise

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 8, 3], [7, 5, 6], [4, 2, 9] ], [7, 8, 9]),
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [-1, 2, 3], [-2, 5, 6], [-3, 2, 9] ], [-1, 5, 9])
    ])
def test_daily_max(test, expected):
    """Tests that max function works for an array of positive integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [1, 8, 3], [7, 5, 6], [4, 2, 9] ], [1, 2, 3]),
        ([ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        ([ [-1, 2, 3], [-2, 5, 6], [-3, 2, 9] ], [-3, 2, 3])
    ])
def test_daily_min(test, expected):
    """Tests that min function works for an array of positive integers."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

def test_daily_min_string():
    """Tests for TypeError when passing strings."""

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]]),
        ([[np.nan, 2, 3], [9, 7, 6], [-1, 2, np.nan]], [[0., 0.67, 1], [1, 0.78, 0.67], [0, 1, 0]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)

"""
@pytest.mark.parametrize(
    "func, ""test, expected",
    [
        (daily_mean, [ [1, 8, 3], [7, 5, 6], [4, 2, 9] ], [7, 8, 9]),
        (daily_max, [ [0, 0, 0], [0, 0, 0], [0, 0, 0] ], [0, 0, 0]),
        (func_min, [ [-1, 2, 3], [-2, 5, 6], [-3, 2, 9] ], [-1, 5, 9])
    ])
def test_daily_max(func, test, expected):
    Tests that max function works for an array of positive integers.
    npt.assert_array_equal(func(np.array(test)), np.array(expected))
"""