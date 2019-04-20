import src.obstacle as obs
import theano.tensor as tt
import numpy as np


def test_gm_distance_field():

    x = np.random.rand(10, 3)
    mu = np.random.rand(4, 3)
    cov = np.random.rand(4, 3, 3)
    cov += np.transpose(cov, (0, 2, 1))
    prec = np.linalg.inv(cov)

    npd = obs.np_gm_obstacle_cost(x, mu, prec)

    ttx = tt.dmatrix('x')
    ttmu = tt.constant(mu, 'mu')
    ttprec = tt.constant(prec, 'prec')
    f = obs.f_gm_distance_field(ttx, ttmu, ttprec)
    thd = f(x)

    assert np.allclose(npd, thd)
