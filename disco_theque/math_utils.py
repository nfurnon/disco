import numpy as np


def floor_to_multiple(num, div):
    """ Returns the highest multiple of div that is lower than num.
    e.g. floor_to_multiple(102, 10) = 100
         floor_to_multiple(65, 8) = 64
    """
    return np.int(num - (num % div))


def round_to_base(x, base=1):
    """
    Round to next integer, with step 'base'.
    e.g. round_to_base(109.56, 5) = 110
         round_to_base(109.56, 4) = 108
         round_to_base(56, 10) = 60
    """
    return base * np.round(x / base)


def db2lin(x, exp=1):
    """Decibel to linear converter
    exp = 1: Decibel to power
    exp = 2: Decibel to magnitude
    """
    exp_ = exp*10
    y = 10**(x/exp_)
    return y


def lin2db(x):
    """Converter to decibel values
    """
    return 10*np.log10(x)


def cart2pol(x, y):
    """
    Returns the coordinate in polar system
    :param x:
    :param y:
    :return: r:         radius
    :return: theta      in rad, the angle
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Variant (only works for scalars):
    # import cmath
    # a_z = cmath.polar(complex(x, y))
    return r, theta


def pol2cart(r, theta):
    """
    Returns the polar coordinate in  cartesian system
    :param r:           radius
    :param theta:       in rad, the angle
    :return: x, y:      cartesian coordinates
    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def my_mse(x, y):
    """
    Compute MSE between two arrays following keras logic but without the weighting (simple mean)
    :param x:       (array) estimated signal
    :param y:       (array) reference
    :return:        mean(abs(x-y)**2))
    """
    return np.mean(np.mean((x - y)**2, axis=-1))


def next_pow_2(x):
    """Calculates the next power of 2 of a number.

    Parameters
    ----------
    x : float
        Number for which to calculate the next power of 2.

    Returns
    -------
    int

    Copyright https://github.com/achabotl/pambox/blob/develop/pambox/utils.py

    """
    return int(pow(2, np.ceil(np.log2(x))))


class WelfordsOnlineAlgorithm:
    """
    Computes the statistics of a database without loading all the data at one.
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm. We adapt it
    in the case of 2-D data.
    """
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.mean = np.zeros(self.feature_dim)
        self.m2 = np.zeros(self.feature_dim)
        self.std = np.zeros(self.feature_dim)
        self.count = 0

    def update_stats(self, data):
        """
        Update mean and std given new samples in data.
        :param data:        Assumed (feature_dim x n_frames)
        """
        data = np.array(data)
        assert data.shape[0] == self.feature_dim, \
            "`data` should have {} features, got {}".format(str(self.feature_dim), str(data.shape[0]))
        for frame in data.T:
            self.count += 1
            delta = frame - self.mean
            # Update mean
            self.mean += delta / self.count
            # Update variance
            delta2 = frame - self.mean
            self.m2 += delta * delta2
            self.std = np.sqrt(self.m2 / self.count)


