import numpy as np
import theano as th
import sympy as sm
import theano.tensor as tt
from typing import *


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


def th_smoothness(q: tt.TensorVariable=None, w: tt.TensorConstant=None):
    if q is None:
        q = tt.dmatrix("q")  # type: tt.TensorVariable

    # Backward differences.
    dd = abs(q[1:] - q[:-1])
    if w is not None:
        dd = dd * w.dimshuffle('x', 0)
    y = .5 * tt.tensordot(dd, dd)

    return y, q


def ffp_smoothness(q: tt.TensorVariable=None):
    y, q = th_smoothness(q)

    f = th.function(inputs=[q], outputs=y)

    dfdq = th.grad(cost=y, wrt=q)
    fp = th.function(inputs=[q], outputs=dfdq)

    return f, fp, q


def th_obstacle(q: tt.TensorVariable, u: tt.TensorConstant,
                xf: Callable[[tt.TensorVariable, tt.TensorConstant], tt.Tensor],
                cf: Callable[[tt.Tensor], tt.Tensor]):
    """

    :param q: The configurations over the trajectory, in order of time.
    :param u: Points on the discretized robot body.
    :param xf: A function mapping workspace config and body to workspace.
    :param cf: A function mapping workspace to obstacle costs.
    """

    # Pass our configuration and robot body to get workspace coords.
    xqu = xf(q, u)  # .shape == (Q, U, D)

    # Pass our workspace coords to get our obstacle cost function.
    cxqu = cf(xqu)  # .shape == (Q, U)

    # Average of adjacent t for each robot element.
    cxqu_cd = .5 * (cxqu[1:, :] + cxqu[:-1, :])  # .shape == (Q-1, U)

    # Backward differences...
    xqu_bd = xqu[1:] - xqu[:-1]  # .shape == (Q-1, U, D)

    return tt.sum(cxqu_cd.dimshuffle(0, 1, 'x') * xqu_bd), q


def ffp_obstacle(q, *args, **kwargs):
    y, q = th_obstacle(q, *args, **kwargs)

    f = th.function(inputs=[q], outputs=y)

    dfdq = th.grad(cost=y, wrt=q)
    fp = th.function(inputs=[q], outputs=dfdq)

    return f, fp, q