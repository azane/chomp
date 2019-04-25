import numpy as np
import theano as th
import theano.tensor as tt
from src.th_matmul import matmul

# Code defining obstacle cost functions, distance fields, etc.


# <Direct Gaussian Mixture Obstacle Cost>
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

    # th.config.compute_test_value = 'warn'
    # x.tag.test_value = np.random.rand(10, 3)

    mux = mu.dimshuffle(0, 'x', 1) - x.dimshuffle('x', 0, 1)

    mux = mux.dimshuffle(0, 1, 2, 'x')  # .shape == K, N, D, 1
    mux_T = mux.dimshuffle(0, 1, 3, 2)  # .shape == K, N, 1, D

    prec = prec.dimshuffle(0, 'x', 1, 2)  # .shape == K, 1, D, D

    mux_prec = matmul(mux_T, prec)  # (K, N, 1, D) * (K, N, D, D) == (K, N, 1, D)
    mux_prec_mux = matmul(mux_prec, mux)  # (K, N, 1, D) * (K, N, D, 1) == (K, N, 1, 1)

    return tt.exp(-.5*mux_prec_mux).dimshuffle(1, 0) # .shape == N, K


def th_gm_obstacle_cost_wrap(mu: tt.TensorConstant, prec: tt.TensorConstant):

    def wrap(x: tt.TensorVariable):
        # Flatten all but the last dimension.
        x_ = x.reshape(shape=(-1, x.shape[-1]), ndim=2)  # (Q, U, D) => (Q*U, D)

        # Get result and sum over all the obstacle gradients, then normalize for the number of obstacles.
        res = th_gm_obstacle_cost(x_, mu, prec)  # .shape == (Q*U, K)
        res = tt.sum(res, axis=-1) / res.shape[-1]  # .shape == (Q*U,)

        # Restore original shape, but without workspace dimension axis (the last one)
        return res.reshape(shape=(x.shape[:-1]), ndim=x.ndim-1)  # .shape == (Q, U)

    return wrap


def th_gm_closest_obstacle_cost_wrap(mu: tt.TensorConstant, prec: tt.TensorConstant):

    def wrap(x: tt.TensorVariable):
        # Flatten all but the last dimension.
        x_ = x.reshape(shape=(-1, x.shape[-1]), ndim=2)  # (Q, U, D) => (Q*U, D)

        # Get result and select closest obstacle (one with the "worst" objective value.
        res = th_gm_obstacle_cost(x_, mu, prec)  # .shape == (Q*U, K)
        res = tt.max(res, axis=-1)  # .shape == (Q*U,)

        # Restore original shape, but without workspace dimension axis (the last one)
        return res.reshape(shape=(x.shape[:-1]), ndim=x.ndim-1)  # .shape == (Q, U)

    return wrap


def f_gm_obstacle_cost(x: tt.TensorVariable, mu: tt.TensorConstant, prec: tt.TensorConstant):
    return th.function([x], th_gm_obstacle_cost(x, mu, prec))
# </Direct Gaussian Mixture Obstacle Cost>


# <Elliptical Obstacle Distance Field>
# This is essentially a generalization of the above, but we compute the cost in actual distance space.
# Further, we compute the point of closest approach along a line between two points, and use that distance.

def np_el_backproject_all(x, mu, Ainv):
    # Project the points in x back to each ellipse. This essentially moves
    #  from an elliptical gaussian to a spherical gaussian in std units.
    # Ainv is from inv(cholesky(cov))
    assert x.shape[1] == mu.shape[1] == Ainv.shape[1] == Ainv.shape[2]

    x = x[:, None, :] - mu[None, :, :]  # (N, K, D)
    x = np.matmul(Ainv[None, ...], x[..., None])  # (1, K, D, D) . (N, K, D, 1)
    x = x.squeeze(-1)  # (N, K, D)
    return x


def np_el_nearestd(x1, x2, mu, Ainv):
    # Compute the distance of closest approach between each pair of points in
    #  x1 and x2. Return only the distance to the closest obstacle along each line.
    # This distance is in standard deviation units.

    # Other assertions are handled in np_el_backproject
    D = x1.shape[1]

    # First, backproject the points into each ellipse's spherical space.
    x1g = np_el_backproject_all(x1, mu, Ainv)  # (N, K, D)
    x2g = np_el_backproject_all(x2, mu, Ainv)

    # From http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    # Our x0 is simply the origin here (because we projected back to a zero mean std=1 gaussian).

    x1gf = x1g.reshape(-1, D)
    x2gf = x2g.reshape(-1, D)

    diff = x2gf - x1gf
    num = -np.matmul(x1gf[..., None, :], diff[..., :, None]).squeeze()
    # Squared euc norm.
    den = np.matmul(diff[..., None, :], diff[..., :, None]).squeeze()

    t = num / den

    # If t is negative, or greater than diff, this point is out of bounds.
    tneg = t < 0
    tbig = t > np.sqrt(den)
    tout = np.logical_or(tneg, tbig)
    t[tout] = np.inf

    d = np.linalg.norm(x1gf + diff*t[:, None], axis=1)
    d = d.reshape(x1g.shape[:-1])  # (N, K)
    return np.min(d, axis=1)

# TODO make theano versions of the above and use as our obstacle obstacle distance computation for CHOMP.

# <Elliptical Obstacle Distance Field>
