import theano.tensor as tt
import numpy as np
import src.obstacle as obs
import src.objective as obj
import src.kinematics as kn
from typing import *

U = 20
D = 3
Q = 100
BBOX = 60.


def get_L_3d_const_robot() -> tt.TensorConstant:
    # Return an L-shaped robot, but not randomly.
    U_ = int(U/2)
    x = np.ones((U_, D)) * np.linspace(0, 8, U_)[:, None]
    y = np.copy(x)
    x[:, 0] = 0
    y[:, 1] = 0
    xyz = np.vstack((x, y))
    xyz[:, 2] = 0

    return tt.constant(xyz)


# Clear path check. Wrapper so we only compute Ainv once per set of obstacles.
def path_clear_wrap(mu, cov, f_xf):

    Ainv = np.linalg.inv(np.linalg.cholesky(cov))

    def wrap(q):
        # Clear if the path of closest approach between all adjacent qpath members
        #  is outside of the obstacle closest to that path.
        xx = f_xf(q)
        x1 = xx[1:]
        x2 = xx[:-1]
        d = obs.np_el_nearestd(x1=x1.reshape(-1, D), x2=x2.reshape(-1, D), mu=mu, Ainv=Ainv)
        # Clear if no point is within 2.1 stdevs of an ellipse.
        col = d < 2.1
        return not np.any(col)

    return wrap


def solve_chomp(qinit, f_obj: Callable, fp_obj: Callable, path_clear: Callable, maxiter=75)\
        -> Generator[Tuple[np.ndarray, float, int], None, bool]:

    qn = np.copy(qinit)

    K_ = obj.slow_fdiff_1(len(qn) - 2)
    Ainv = np.linalg.inv(K_.T.dot(K_) + np.diag(np.ones(K_.shape[1]) * 0.001))
    step = np.array([0.01, 0.01, 0.01, 0.0005, 0.0005, 0.0005])

    last_obj = f_obj(qn)

    for n in range(maxiter):

        qnp = fp_obj(qn)
        qn[1:-1] -= step[None, :] * Ainv.dot(qnp[1:-1])

        # If any angle-axes of our 6dof angle-axes are moving toward zero,
        #  adjust them all away.
        if np.any(np.less(qn[:, 3:].sum(axis=1), .2)):
            qn = kn.unzero_6dof(qn)

        this_obj = f_obj(qn)
        if this_obj > last_obj:
            step *= .4
        else:
            step *= 1. / .9
        last_obj = this_obj

        yield qn, this_obj, n

        if path_clear(qn):
            return True

    return False


def get_6dof_straight_path():
    # The boundary conditions and initial path, init to a straight line.
    qstart = np.ones(3) * -BBOX
    qend = -qstart
    qvec = (qend - qstart) / Q
    qarange = np.arange(Q)[:, None]
    qpath = np.hstack((qarange,) * D) * qvec + qstart
    qpath = np.hstack((qpath, np.ones((Q, 3)) * 2 * np.pi))

    return qpath


def get_spheres(k: int, s: float):

    mu = np.random.uniform(-BBOX+s, BBOX-s, size=(k, D))
    cov = np.tile(np.eye(D)*s**2., k).T.reshape(k, D, D)

    return mu, cov
