import numpy as np
import scipy.linalg
import sys
from disco_theque.misc_utils import truncated_eye

eps = sys.float_info.epsilon
eta = 1e6


def get_filter_type(filtre):
    """From filter_name, return which type of GEVD-based MWF [1] is expected in the inter_filter function, and, if
    relevant, which rank constraint.
    :return - filtre    (str) gevd, mwf
            - rank      (int) rank of the constrained correlation matrix
    [1] R. Serizel, M. Moonen, B. Van Dijk and J. Wouters,
        "Low-rank Approximation Based Multichannel Wiener Filter Algorithms for Noise Reduction with Application in
        Cochlear Implants," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, April 2014.
    """
    if 'gevd' in filtre:
        if '-' in filtre:
            rank = int(filtre.split('-')[0][-1])
        else:
            rank = 'Full'
        filtre = 'gevd'
    else:
        rank = None

    return filtre, rank


def intern_filter(Rxx, Rnn, mu=1, type='r1-mwf', rank='Full'):
    """
    Computes a filter according to references [1] (SDW-MWF) or [2] (GEVD-MWF).

    :param Rxx: Speech covariance matrix
    :param Rnn: Noise covariance matrix
    :param mu: Speech distortion constant [default 1]
    :param type: (string) Type of filter to compute (SDW-MWF (see [1], equation (4)) or GEVD (see [2])). [default 'mwf']
    :param rank: ('Full' or 1) Rank-1 approximation of Rxx ? (see [2]). [default 'Full']
    :return:    - Wint: Filter coefficients
                - t1: (for GEVD case) vector selecting signals in GEVD fashion, to get correct reference signal
    """
    t1 = np.squeeze(np.vstack((1, np.zeros((np.shape(Rxx)[0] - 1, 1)))))    # Default is e1, selecting first column
    sort_index = None
    if type == 'r1-mwf':
        # # ----- Make Rxx rank 1  -----
        D, X = np.linalg.eig(Rxx)
        D = np.real(D)  # Must be real (imaginary part due to numerical noise)
        Dmax, maxind = D.max(), D.argmax()  # Find maximal eigenvalue
        Rxx = np.outer(np.abs(Dmax) * X[:, maxind],
                       np.conjugate(X[:, maxind]).T)  # Rxx is assumed to be rank 1
        # -----------------------------
        P = np.linalg.lstsq(Rnn, Rxx, rcond=None)[0]
        Wint = 1 / (mu + np.trace(P)) * P[:, 0]  # Rank1-SDWMWF (see [1])

    elif type == 'gevd':
        # TODO: inquire wether scipy.linalg.eig is much slower than scipy.linag.eigh
        D, Q = scipy.linalg.eig(Rxx, Rnn)               # GEV decomposition of Rnn^(-1)*Rss
        D = np.maximum(D,
                       eps * np.ones(np.shape(D)))      # Prevent negative eigenvalues
        D = np.minimum(D,
                       eta * np.ones(np.shape(D)))      # Prevent infinite eigenvalues
        sort_index = np.argsort(D)                      # Sort the array to put GEV in descending order in the diagonal
        D = np.diag(D[sort_index[::-1]])                # Diagonal matrix of descending-order sorted GEV
        Q = Q[:, sort_index[::-1]]                      # Sorted matrix of generalized eigenvectors
        if rank != 'full':                              # Rank-1 matrix of GEVD;
            D[rank:, :] = 0                             # Force zero values for all GEV but the highest
        # Filter
        Wint = np.matmul(Q,
                         np.matmul(D,
                                   np.matmul(np.linalg.inv(D + mu * np.eye(len(D))),
                                             np.linalg.inv(Q))))[:, 0]
        t1 = Q[:, 0] * np.linalg.inv(Q)[0, 0]
    elif type == 'mwf':
        P = np.linalg.lstsq(Rnn + Rxx, Rxx, rcond=None)[0]
        Wint = P[:, 0]

    else:
        raise AttributeError('Unknown filter reference')

    return Wint, (t1, sort_index)


def spatial_correlation_matrix(Rxx, x, lambda_cor=0.95, M=None):
    """
    Return spatial correlation matrix computed as exponentially smoothing of :
            - if M is None: x*x.T
                            so Rxx = lambda * Rxx + (1 - lambda)x*x.T
              x should then be an estimation of the signal of which one wants the Rxx

            - if M is not None: M*x*x.T
              x is then the mixture
    :param Rxx:             Previous estimation of Rxx
    :param x:               Signal (estimation of noise/speech if M is none; mixture otherwise)
    :param lambda_cor:      Smoothing parameter
    :param M:               Mask. If None, x is the estimation of the signal of which one wants the Rxx.
    :return: Rxx            Current eximation of Rxx
    """
    if M is None:
        Rxx = lambda_cor * Rxx + (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    else:
        Rxx = lambda_cor * Rxx + M * (1 - lambda_cor) * np.outer(x, np.conjugate(x).T)
    return Rxx
