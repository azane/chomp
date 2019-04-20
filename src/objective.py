import numpy as np
import theano as th
import sympy as sm
import theano.tensor as tt


def slow_fdiff_1(n: int) -> np.ndarray:
    K = np.diag(np.ones(n) * -1, 0)
    K += np.diag(np.ones(n - 1) * 1, 1)
    K = np.vstack((np.zeros(n), K))
    K[0, 0] = 1.
    K[-1, -1] = -1.
    return K


def slow_naive_prior(q: np.ndarray) -> float:
    assert q.ndim == 2
    tot = 0.
    for i, qq in enumerate(q[:-1]):
        dd = (q[i+1] - qq)
        tot += np.inner(dd, dd)
    return .5 * tot


def slow_fdmat_prior(q: np.ndarray) -> float:
    assert q.ndim == 2

    # Set up boundary condition vector.
    e = np.zeros(q[1:].shape)
    e[0] = -q[0]
    e[-1] = q[-1]

    # Difference over all but boundaries.
    K = slow_fdiff_1(len(q)-2)
    dd = K.dot(q[1:-1]) + e
    return .5 * np.tensordot(dd, dd)


def th_smoothness(q: tt.TensorVariable=None):
    if q is None:
        q = tt.dmatrix("q")  # type: tt.TensorVariable

    dd = q[1:] - q[:-1]
    y = .5 * tt.tensordot(dd, dd)

    return y, q


def ffp_smoothness(q: tt.TensorVariable=None):
    y, q = th_smoothness(q)

    f = th.function(inputs=[q], outputs=y)

    dfdq = th.grad(cost=y, wrt=q)
    fp = th.function(inputs=[q], outputs=dfdq)

    return f, fp, q




