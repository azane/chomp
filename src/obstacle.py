import numpy as np
import theano as th
import theano.tensor as tt
from src.th_matmul import matmul

def np_gm_obstacle_cost(x: np.ndarray, mu: np.ndarray, prec: np.ndarray):

    # Shapes, N = num points, K = num gaussians, D = dimensionality.
    mux = mu[:, None, :] - x[None, :, :]  # .shape == K, N, D

    # Column matrix in last 2 dims.
    mux = mux[:, :, :, None]  # .shape == K, N, D, 1
    # Row matrix in last 2 dims.
    mux_T = np.transpose(mux, (0, 1, 3, 2))  # .shape == K, N, 1, D

    prec = prec[:, None, :, :]  # .shape == K, 1, D, D

    mux_prec = np.einsum('knij,knji->kni', mux_T, prec)  # .shape == K, N, D
    mux_prec = mux_prec[:, :, None, :]  # .shape == K, N, 1, D
    mux_prec_mux = np.einsum('knij,knji->nk', mux_prec, mux)  # .shape == N, K

    return np.exp(-.5*mux_prec_mux)


def th_gm_obstacle_cost(x: tt.TensorVariable, mu: tt.TensorConstant, prec: tt.TensorConstant):
    assert mu.ndim == 2
    assert x.ndim == 2
    assert prec.ndim == 3

    th.config.compute_test_value = 'warn'

    x.tag.test_value = np.random.rand(10, 3)

    mux = mu.dimshuffle(0, 'x', 1) - x.dimshuffle('x', 0, 1)

    mux = mux.dimshuffle(0, 1, 2, 'x')  # .shape == K, N, D, 1
    mux_T = mux.dimshuffle(0, 1, 3, 2)  # .shape == K, N, 1, D

    prec = prec.dimshuffle(0, 'x', 1, 2)  # .shape == K, 1, D, D

    mux_prec = matmul(mux_T, prec)  # (K, N, 1, D) * (K, N, D, D) == (K, N, 1, D)
    mux_prec_mux = matmul(mux_prec, mux)  # (K, N, 1, D) * (K, N, D, 1) == (K, N, 1, 1)

    return tt.exp(-.5*mux_prec_mux).dimshuffle(1, 0) # .shape == N, K


def f_gm_distance_field(x: tt.TensorVariable, mu: tt.TensorConstant, prec: tt.TensorConstant):
    return th.function([x], th_gm_obstacle_cost(x, mu, prec))
