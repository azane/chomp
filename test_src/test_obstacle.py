import src.obstacle as obs
import theano.tensor as tt
import numpy as np
import time


def test_gm_distance_field():

    x = np.random.rand(100, 3)
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

    # t1 = []
    # for _ in range(1000):
    #     t = time.time()
    #     obs.np_gm_obstacle_cost(x, mu, prec)
    #     t1.append(time.time() - t)
    # t2 = []
    # for _ in range(1000):
    #     t = time.time()
    #     f(x)
    #     t2.append(time.time() - t)
    # print("np", np.mean(t1), np.std(t1))
    # print("th", np.mean(t2), np.std(t2))

    assert np.allclose(npd, thd)
