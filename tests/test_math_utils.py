import pytest

from context import disco_theque
import disco_theque.math_utils as dmath


@pytest.mark.parametrize('num, div, expected', (
    (102, 10, 100),
    (65, 8, 64),
    (33, 5, 30),
    )
)
def test_floor_to_multiple(num, div, expected):
    assert dmath.floor_to_multiple(num, div) == expected
