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

