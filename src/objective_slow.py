import numpy as np
import theano as th
import sympy as sm


def slow_fdiff_1(n: int) -> np.ndarray:
    K = np.diag(np.ones(n), 0)
    K += np.diag(np.ones(n - 1) * -1, 1)
    K = np.vstack((np.zeros(n), K))
    K[0, 0] = 1.
    K[-1, -1] = -1.
    return K


def slow_naive_prior(q: np.ndarray, dt: float) -> float:
    assert q.ndim == 2
    tot = 0.
    hh = []
    for i, qq in enumerate(q[:-1]):
        dd = (q[i+1] - qq) / dt
        # tot += np.inner(dd, dd)
        hh.append(dd)
    return np.array(hh)
    # return .5 * tot


def slow_fdmat_prior(q: np.ndarray, dt: float) -> float:
    assert q.ndim == 2

    # Set up boundary condition vector.
    e = np.zeros(q[1:].shape)
    e[0] = -q[0]
    e[-1] = q[-1]

    # Difference over all but boundaries.
    K = slow_fdiff_1(len(q)-2) * dt
    dd = K.dot(q[1:-1]) + e
    return dd
    # return .5 * np.tensordot(dd, dd)


def slow_quad_prior(q: np.ndarray, dt: float) -> float:
    assert q.ndim == 2

    e = np.copy(q[:, 0])
    e[1:-1] = 0

    K = slow_fdiff_1(len(q)) * dt
    A = K.T.dot(K)
    b = K.dot(e)
    c = .5 * e.dot(e)

    return .5 * q.T.dot(A).dot(q) + q.T.dot(b) + c
