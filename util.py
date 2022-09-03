import numpy as np
from scipy.spatial.distance import cdist


def squared_maha(x, y=None, Q=None):
    # x: (Nx, D)
    # y: (Ny, D)
    # A: (n, D, D)
    if y is None:
        D_xy = x.copy()
    else:
        D_xy = x[:, None] - y[None, :]  # (Nx, Ny, D)
    if Q is None:
        Q = np.eye(x.shape[1])
    return D_xy @ Q @ D_xy.swapaxes(-1, -2)


# will be deprecated
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