import numpy as np
import theano.tensor as tt
import sys
import os

D = 3
U = 20
Q = 50

# TODO hack yay circular imports...
dpath = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_L_3d_const_robot() -> tt.TensorConstant:
    # Return an L-shaped robot, but not randomly.
    U_ = 10
    x = np.ones((U_, 2))* np.linspace(0, 8, U_)[:, None]
    y = np.copy(x)
    x[:, 0] = 0
    y[:, 1] = 0

    return


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
    # TODO HACK. This is weird...yay quick hax.
    if len(sys.argv) == 2:
        if sys.argv[1] == 'u':
            pass
        else:
            mu = np.load(os.path.join(dpath, f"mu{sys.argv[1]}.npy"))
            cov = np.load(os.path.join(dpath, f"cov{sys.argv[1]}.npy"))
            return mu, cov

    if len(sys.argv) == 2:
        K = int(20)
        scale = 8.
        mu = np.random.uniform(-50., 50., size=(K, D))
    else:
        K = int(15)
        scale = 10.
        mu = np.random.normal(loc=0., scale=27., size=(K, D))

    dd = np.random.normal(loc=0., scale=scale, size=(K, D, 7))
    cov = []
    for x in dd:
        cov.append(np.cov(x))
    cov = np.array(cov)
    return mu, cov