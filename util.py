import numpy as np
from scipy.spatial.distance import cdist


def smaha(a, b=None, Q=None):
    # squared mahalanobis distance with some additional features
    if b is None:
        # it is not a distance! return: a @ Q @ a.T
        b = np.zeros((1, a.shape[1]))
    else:
        assert a.shape[1] == b.shape[1]

    if Q is None:
        Q = np.eye(a.shape[1])
    else:
        pass

    return np.square(cdist(a, b, metric='mahalanobis', VI=Q))