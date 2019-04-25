import numpy as np
import theano.tensor as tt

D = 3
U = 20
Q = 50


def get_L_3d_robot() -> tt.TensorConstant:

    assert U % 2 == 0

    # Make a weird L shaped robot.
    Lsize = 5.
    lx = np.random.multivariate_normal([Lsize, 0., 0.],
                                       [[Lsize ** 2., 0., 0.],
                                        [0., .1, 0.],
                                        [0., 0., .1]],
                                       int(U/2))
    ly = np.random.multivariate_normal([0., Lsize, 0.],
                                       [[.1, 0., 0.],
                                        [0., Lsize ** 2., 0.],
                                        [0., 0., .1]],
                                       int(U / 2))
    u = tt.constant(np.vstack((lx, ly)))
    return u


def get_6dof_straight_path():
    # The boundary conditions and initial path, init to a straight line.
    qstart = np.ones(3) * -58.
    qend = -qstart
    qvec = (qend - qstart) / Q
    qarange = np.arange(Q)[:, None]
    qpath = np.hstack((qarange,) * D) * qvec + qstart
    qpath = np.hstack((qpath, np.ones((Q, 3)) * 2 * np.pi))

    return qpath


def get_gm_obstacle_field():
    K = int(15)
    mu = np.random.normal(loc=0., scale=27., size=(K, D))
    dd = np.random.normal(loc=0., scale=10., size=(K, D, 7))
    cov = []
    for x in dd:
        cov.append(np.cov(x))
    cov = np.array(cov)
    return mu, cov