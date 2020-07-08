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


@pytest.mark.parametrize('x, y, expected', (
    (1, 1, (math.sqrt(2), math.pi/4)),
    (2, -2, (math.sqrt(8), -math.pi/4)),
    (-1, 1, (math.sqrt(2), 3*math.pi/4)),
    (np.array([1, 1, -1, -1]),
     np.array([1, -1, 1, -1]),
     (math.sqrt(2)*np.ones(4), math.pi/4*np.array([1, -1, 3, -3]))),
    ))
def test_cart2pol(x, y, expected):
    result = dmath.cart2pol(x, y)
    npt.assert_almost_equal(result, expected)


@pytest.mark.parametrize('r, theta, expected', (
    (math.sqrt(2), math.pi/4, (1, 1)),
    (math.sqrt(2), 7*math.pi/4, (1, -1)),
    (math.sqrt(2)*np.ones(4),
     math.pi/4*np.array([1, -1, 3, -3]),
     (np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]))),
    ))
def test_pol2cart(r, theta, expected):
    result = dmath.pol2cart(r, theta)
    npt.assert_almost_equal(result, expected)


@pytest.mark.parametrize('x, y, expected', (
    (np.ones((4, 100)), np.ones((4, 100)), 0),
    (np.ones((4, 100)), np.zeros((4, 100)), 1),
    ))
def test_my_mse(x, y, expected):
    assert dmath.my_mse(x, y) == expected


@pytest.mark.parametrize('x, expected', (
    (3, 4),
    (120, 128),
    ))
def test_next_pow_2(x, expected):
    assert dmath.next_pow_2(x) == expected


class TestWelfordsOnlineAlgorithm:
    @pytest.fixture(scope='module')
    def feature_dim(self):
        return 2

    @pytest.fixture
    def stats(self, feature_dim):
        return dmath.WelfordsOnlineAlgorithm(feature_dim)

    def test_init(self, stats, feature_dim):
        assert stats.feature_dim == feature_dim
        npt.assert_almost_equal(stats.mean, np.zeros(feature_dim))
        npt.assert_almost_equal(stats.m2, np.zeros(feature_dim))
        npt.assert_almost_equal(stats.std, np.zeros(feature_dim))
        assert stats.count == 0

    def test_update_stats(self, stats, feature_dim):
        data = [np.random.randn(feature_dim, 100), np.random.randn(feature_dim, 400)]

        stats.update_stats(data[0])
        npt.assert_almost_equal(stats.mean, data[0].mean(axis=-1))
        npt.assert_almost_equal(stats.std, data[0].std(axis=-1))
        assert stats.count == 100

        stats.update_stats(data[1])
        whole_data = np.concatenate(data, axis=-1)
        npt.assert_almost_equal(stats.mean, whole_data.mean(axis=-1))
        npt.assert_almost_equal(stats.std, whole_data.std(axis=-1))
        assert stats.count == 500

    def test_update_stats_wrong_dim(self, stats, feature_dim):
        data = np.zeros((feature_dim + 1, 2))
        with pytest.raises(AssertionError) as err:
            stats.update_stats(data)
        assert f'`data` should have {feature_dim} features, got {feature_dim+1}' in str(err)
