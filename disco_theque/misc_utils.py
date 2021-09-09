import yaml
import numpy as np
import string
import random


def get_node_from_channel(ch, arr_geo):
    """
    Return node corresponding to microphone number ch.
:param ch       channel or microphone number (starts at 0)
    :param arr_geo  geometry of the array. Array of number of microphones per node
    :return         node corresponding to input channel
    """
    mics_cum = np.cumsum(arr_geo)
    return list(ch < mics_cum).index(True)      # First index where mic is smaller than the cumulated
                                                # microphones of the array


def find_unmatched_dim(arr1, arr2):
    """
    Find the dimensions of aar1 and arr2 that do not have the same length. arr1 and arr2 should have the same number of
    dimensions.
    :param arr1:        First array
    :param arr2:        Second array
    :return dims:       Array of dimension indices where arr1 and arr2 shapes do not match
    """
    return (np.array(arr1.shape) - np.array(arr2.shape) != 0).nonzero()


def concatenate_dicts(dict_list):
    """
    Concatenate dictionaries that share the same keys
    :param dict_list:   List containing of dictionaries to concatenate
    :return cct:        Concatenation of input dictionaries
    """

    cct = dict_list[0].copy()
    for dictry in dict_list[1:]:
        for k in cct.keys():
            cct_axis = np.array(find_unmatched_dim(cct[k], dictry[k]))
            if cct_axis.size:
                cct[k] = np.concatenate((cct[k], dictry[k]), axis=cct_axis[0][0])
            else:
                cct[k] = np.concatenate((cct[k], dictry[k]), axis=0)

    return cct


def repeat_matrix(a, nb_repeats):
    """
    Stack a nb_repeats times with itself on the third axis.
    :param a:               Matrix to repeat. Should be 2D.
    :param nb_repeats:      Nb of times the matrix should be repeated
    :return:                A tile of a over the third axis
    """
    b = np.tile(a, (1, nb_repeats)).reshape((a.shape[0], a.shape[1], -1), order='F')
    return b


def truncated_eye(N, j, k=0):
    """
    Create a NxN matrix with k consecutive ones in the diagonal.
    :param N:   (int) Dimension of output matrix
    :param j:   (int) Number of ones in the diagonal
    :param k:   (int) Diagonal in question (k>0 shifts the diagonal to a sub-diagonal)
    :return: A truncated eye matrix
    """
    v1 = np.ones((j, ))
    v0 = np.zeros((N - j, ))

    return np.diag(np.concatenate((v1, v0), axis=0), k=k)


def trim_2d_array(mat, axis=0, trim='fb'):
    """
    Trim the leading and/or trailing zeros of a 2D array.
    :param mat:     (array) Array to trim
    :param axis:    (int)   Axis along which to trim
    :param trim:    (str)   'b', 'f', 'fb': trim trailing / leading / trailing and leading zeros ?
    :return:        (array) of reduced shape, without the leading and/or trailing zeros along axis

    credits: Eddmik on https://stackoverflow.com/a/50699907
    """
    assert trim in ['f', 'b', 'fb'], "`trim` can only be 'f', 'b' or 'fb'."
    spots = ~(mat == 0).all(axis=axis)
    inv_spots = spots[::-1]
    if 'f' in trim:
        start_idx = np.argmax(spots == True)
    else:
        start_idx = 0
    if 'b' in trim:
        end_idx = len(inv_spots) - np.argmax(inv_spots == True)
    else:
        end_idx = mat.shape[1 - axis]

    if axis:
        return mat[start_idx:end_idx, :]
    else:
        return mat[:, start_idx:end_idx]


def bar_data(x_edges, x, y):
    """Gather y data into bins in order to plot.bar it; return the average and CI of each bin."""
    from disco_theque.metrics import ci_wp
    y_out = [[] for _ in range(len(x_edges))]
    for i in range(len(x)):
        i_bin = list(x[i] > x_edges).index(False)
        y_out[i_bin].append(y[i])
    means =  [[] for _ in range(len(x_edges))]
    cis =  [[] for _ in range(len(x_edges))]
    for i in range(len(y_out)):
        means[i] = np.nanmean(y_out[i])
        cis[i] = ci_wp(y_out[i])
    return np.array(means), np.array(cis)


def get_random_string(length):
    """Return a string of length `length` containing random characters picked among any ascii upper- and lowercase
    letters or digits.
    Args:
        length (int)    length of the output string (number of characters)
    Returns:
        a string of `length` random characters.
    Credits https://pynative.com/python-generate-random-string/
    """
    all_chars = string.ascii_letters + string.digits
    return ''.join((random.choice(all_chars) for _ in range(length)))


def yaml2dict(yaml_file):
    """
    Very custom. To do: generalize to any number of sub-dicts and names of keys.
    Return the values in `yaml_file` as a dictionnary of integers
    """
    with open(yaml_file) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)

    for k in params.keys():
        params[k] = integerize(params[k])

    return params


def integerize(values):
    """Return values as integers. Useful only if they are a list of strings."""
    if isinstance(values, dict):
        for k in values.keys():
            values[k] = integerize(values[k])
    elif isinstance(values, str):
        try:
            values = np.array(values.split(' '), dtype=int)
        except ValueError:
            if values == 'None':
                values = None
            elif ' ' in values:
                values = values.split(' ')
            else:
                pass

    return values

