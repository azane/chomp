import src.obstacle as obs
import theano.tensor as tt
import theano as th
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


def test_th_np_nearestd():
    # Compare the numpy and theano implementations.

    K = 5
    D = 3
    Q = 100
    U = 10

    mu = np.random.normal(loc=0., scale=3., size=(K, D))
    dd = np.random.normal(loc=0., scale=5., size=(K, D, D * 2))
    cov = []
    for d in dd:
        cov.append(np.cov(d))
    cov = np.array(cov)
    covAinv = np.linalg.inv(np.linalg.cholesky(cov))

    # Compute the numpy version.
    xx = np.random.uniform(-6., 6., size=(Q, U, D))
    x1 = xx[1:]
    x2 = xx[:-1]
    np_d = obs.np_el_nearestd(x1=x1.reshape(-1, D), x2=x2.reshape(-1, D), mu=mu, Ainv=covAinv)

    # Compute the theano version.
    tt_x1 = tt.dmatrix('x1')
    tt_x2 = tt.dmatrix('x2')
    tt_mu = tt.constant(mu)
    tt_covAinv = tt.constant(covAinv)
    th.config.compute_test_value = 'warn'
    tt_x1.tag.test_value = x1.reshape(-1, D)
    tt_x2.tag.test_value = x2.reshape(-1, D)
    tt_f = obs.th_el_nearestd(x1=tt_x1, x2=tt_x2, mu=tt_mu, Ainv=tt_covAinv)
    f = th.function(inputs=[tt_x1, tt_x2], outputs=tt_f, mode=th.compile.FAST_COMPILE)
    th_d = f(x1.reshape(-1, D), x2.reshape(-1, D))

    assert np.allclose(np_d, th_d)

    for _ in range(10):
        xx = np.random.uniform(-6., 6., size=(Q, U, D))
        x1 = xx[1:]
        x2 = xx[:-1]
        np_d = obs.np_el_nearestd(x1=x1.reshape(-1, D), x2=x2.reshape(-1, D), mu=mu, Ainv=covAinv)
        th_d = f(x1.reshape(-1, D), x2.reshape(-1, D))
        assert np.allclose(np_d, th_d)

