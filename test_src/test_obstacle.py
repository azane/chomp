import src.obstacle as obs
import theano.tensor as tt
import numpy as np
import time


# Couple this test with the visualization gm_obstacle_cost.

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
    f = obs.f_gm_obstacle_cost(ttx, ttmu, ttprec)
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


def test_el_backproject():
    K = 1
    D = 3
    mu = np.random.normal(loc=0., scale=3., size=(K, D))
    dd = np.random.normal(loc=0., scale=5., size=(K, D, D*2))
    cov = []
    for d in dd:
        cov.append(np.cov(d))
    cov = np.array(cov)

    # Points in sphere space.
    x_s = np.random.normal(loc=0., scale=10., size=(100, D))

    # Project points out into elliptical space.
    A = np.linalg.cholesky(cov)
    x_e = np.matmul(A[None, ...], x_s[:, None, :, None])
    x_e = x_e.squeeze() + mu

    # Project them back into spherical space using the test function.
    Ainv = np.linalg.inv(A)
    x_s2 = obs.np_el_backproject_all(x_e, mu, Ainv)

    assert np.allclose(x_s, x_s2.squeeze())