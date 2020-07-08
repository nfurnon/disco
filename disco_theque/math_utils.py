import numpy as np


def floor_to_multiple(num, div):
    """Computes the highest multiple of `div` that is lower than `num`.

    Examples:
        >>> floor_to_multiple(102, 10)
        100
        >>> floor_to_multiple(65, 8)
        64

    Args:
        num (int):
        div (int):

    Returns:
        int: Highest multiple of `div` that is lower than `num`

    """
    return np.int(num - (num % div))


def round_to_base(x, base=1):
    """Rounds to next integer, with step `base`.

    Examples:
        >>> round_to_base(109.56, 5)
        110
        >>> round_to_base(108.56, 4)
        108
        >>> round_to_base(56, 10)
        60

    Args:
        x (float): Number
        base (int): Base (Default: 1)

    Returns:
        float: `x` rounded to nearest multiple of `base`

    """
    return base * np.round(x / base)


def db2lin(x, exp=1):
    """Converts decibel to linear scale.

    Note:
        Set `exp` to 1 to convert to power and to 2 to convert to magnitude

    Args:
        x (np.ndarray): Quantity expressed in dB.
        exp (int, optional):  (Default: 1)

    Returns:
        np.ndarray: Power or magnitude expressed on a linear scale

    """
    exp_ = exp*10
    y = 10**(x/exp_)
    return y


def lin2db(x):
    """Converts linear to decibel scale.

    Args:
        x (np.ndarray): Power

    Returns:
        np.ndarray: Power expressed in dB

    """
    return 10*np.log10(x)


def cart2pol(x, y):
    """Converts cartesian to polar coordinates.

    Angles are expressed in radian

    Args:
        x (np.ndarray): Abscissa
        y (np.ndarray): Ordinate

    Returns:
        tuple[np.ndarray, np.ndarray]: radius, angle in radian

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Variant (only works for scalars):
    # import cmath
    # a_z = cmath.polar(complex(x, y))
    return r, theta


def pol2cart(r, theta):
    """Converts polar to cartesian coordinates.

    Args:
        r (np.ndarray): radii
        theta (np.ndarray): angles, in radian

    Returns:
        tuple[np.ndarray, np.ndarray]: Abscissa, ordinate

    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def my_mse(x, y):
    """Computes MSE between two arrays following keras logic but without the weighting (simple mean).

    Written for 2D matrices `x` and `y`

    Args:
        x (np.ndarray): Estimated signal
        y (np.ndarray): Reference

    Returns:
        mean(abs(x-y)**2))

    """
    return np.mean(np.mean((x - y)**2, axis=-1))


def next_pow_2(x):
    """Calculates the next power of 2 of a number.

    Args:
        x (float): Number for which to calculate the next power of 2

    Returns:
        int: Power of 2 closest to and larger than `x`

    """
    return int(pow(2, np.ceil(np.log2(x))))


class WelfordsOnlineAlgorithm:
    """Computes the statistics of a database without loading all the data at one.

    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford%27s_online_algorithm. We adapt it
    in the case of 2-D data.

    Attributes:
        feature_dim (int): Feature dimension
        mean (np.ndarray): Feature mean
        m2 (np.ndarray): Second order central moment
        std (np.ndarray): Feature standard deviation
        count (int): Number of frames used to compute statistics

    """
    def __init__(self, feature_dim):
        """Initializes class instance.

        Args:
            feature_dim (int): Feature dimension
        """
        self.feature_dim = feature_dim
        self.mean = np.zeros(self.feature_dim)
        self.m2 = np.zeros(self.feature_dim)
        self.std = np.zeros(self.feature_dim)
        self.count = 0

    def update_stats(self, data):
        """Update mean and std given new samples in data.

        Args:
            data: Assumed (feature_dim x n_frames)

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

    def quick_update(self, data):
        """Updates stats given new samples.

        This method speeds up computation by vectorizing operations

        Args:
            data (np.ndarray): Data in (`feature_dim` x `n_frames`)

        """
        assert data.shape[0] == self.feature_dim, \
            "`data` should have {} features, got {}".format(self.feature_dim, data.shape[0])

        delta = data - self.mean[:, np.newaxis]
        self.count += data.shape[-1]
        self.mean += delta.sum(axis=-1)/self.count

        delta2 = data - self.mean[:, np.newaxis]
        self.m2 += np.sum(delta2*delta, axis=-1)
        self.std = np.sqrt(self.m2 / self.count)
