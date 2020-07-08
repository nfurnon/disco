import math

import numpy as np
import numpy.testing as npt
import pytest

from context import disco_theque
import disco_theque.math_utils as dmath


@pytest.mark.parametrize('num, div, expected', (
    (102, 10, 100),
    (65, 8, 64),
    (33, 5, 30),
    ))
def test_floor_to_multiple(num, div, expected):
    assert dmath.floor_to_multiple(num, div) == expected


@pytest.mark.parametrize('x, base, expected', (
    (109.56, 5, 110),
    (108.56, 4, 108),
    (56, 10, 60),
    ))
def test_round_to_base(x, base, expected):
    assert dmath.round_to_base(x, base=base) == expected


@pytest.mark.parametrize('x, exp, expected', (
    (0, 1, 1),
    (0, 2, 1),
    (10, 1, 10),
    (10, 2, math.sqrt(10)),
    (np.array([-10, -10]), 1, np.array([0.1, 0.1])),
    ))
def test_db2lin(x, exp, expected):
    result = dmath.db2lin(x, exp=exp)
    npt.assert_equal(result, expected)


@pytest.mark.parametrize('x, expected', (
    (1, 0),
    (10, 10),
    (np.array([100, 1000]), np.array([20, 30])),
    ))
def test_lin2db(x, expected):
    result = dmath.lin2db(x)
    npt.assert_equal(result, expected)
