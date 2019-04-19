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
    for i, qq in enumerate(q[:-1]):
        tot += np.linalg.norm((q[i+1] - qq) / dt)
    return .5 * tot


def slow_fdmat_prior(q: np.ndarray, dt: float) -> float:
    assert q.ndim == 2

    e = np.copy(q)
    e[1:-1, ...] = 0

    K = slow_fdiff_1(len(q)) * dt

    dd = K.dot(q) + e
    return .5 * np.linalg.norm(dd)


def slow_quad_prior(q: np.ndarray, dt: float) -> float:
    assert q.ndim == 2

    e = np.copy(q[:, 0])
    e[1:-1] = 0

    K = slow_fdiff_1(len(q)) * dt
    A = K.T.dot(K)
    b = K.dot(e)
    c = .5 * e.dot(e)

    return .5 * q.T.dot(A).dot(q) + q.T.dot(b) + c
