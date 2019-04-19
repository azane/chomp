import numpy as np
import theano as th
import theano.tensor as tt


# FIXME oops this should be fully broadcasted over mu and x...

def np_gm_distance_field(x: np.ndarray, mu: np.ndarray, prec: np.ndarray):

    # Shapes, N = num points, K = num gaussians, D = dimensionality.
    mux = mu[:, None, :] - x[None, :, :]  # .shape == K, N, D

    # Column matrix in last 2 dims.
    mux = mux[:, :, :, None]  # .shape == K, N, D, 1
    # Row matrix in last 2 dims.
    mux_T = np.transpose(mux, (0, 1, 2, 1))  # .shape == K, N, 1, D

    prec = prec[:, None, :, :]  # .shape == K, 1, D, D

    mux_prec = np.einsum('knij,knji->kni', mux_T, prec)  # .shape == K, N, D
    mux_prec = mux_prec[:, :, :, None]  # .shape == K, N, D, 1
    mux_prec_mux = np.einsum('knij,knij->kni', mux_prec, mux_T)  # .shape == K, N, 1

    return .5 - np.exp(-np.squeeze(mux_prec_mux).T)  # .shape == N, K


def th_gm_distance_field(x: tt.TensorVariable, mu: tt.TensorConstant, prec: tt.TensorConstant):
    """
    A distance field that's .5 - an unnormed mixture of gaussians.
    """
    assert mu.ndim == 2
    assert x.ndim == 2
    assert prec.ndim == 3

    # mu and x should be the same shape.
    mux = (mu - x)

    mux_T = mux.dimshuffle(0, 'x', 1)
    mux_prec = tt.batched_dot(mux_T, prec)
    mux_ = mux.dimshuffle(0, 1, 'x')
    mux_prec_mux = tt.batched_dot(mux_prec, mux_)

    return .5 - tt.exp(tt.squeeze(-mux_prec_mux))


def f_gm_distance_field(x: tt.TensorVariable, mu: tt.TensorConstant, prec: tt.TensorConstant):
    return th.function([x], th_gm_distance_field(x, mu, prec))
