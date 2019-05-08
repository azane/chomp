import theano.tensor as tt
import numpy as np
import src.obstacle as obs
import src.objective as obj
import src.kinematics as kn
from typing import *

U = 20
D = 3
Q = 70
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


def solve_chomp(qinit, f_obj: Callable, fp_obj: Callable, path_clear: Callable,
                maxiter=75, miniter=20)\
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

        if path_clear(qn) and n >= miniter:
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


def chomp_path_from_rrt(q):

    # Bisect the largest gaps until we have enough.

    while len(q) < Q:
        qd = q[1:] - q[:-1]
        # Only count translation part of pose for now for distance. # TODO.
        qd = np.linalg.norm(qd[:, :3], axis=1)
        ql = list(q)
        i = np.argmax(qd)  # type: int

        m = np.zeros(q[i].shape)

        # Midpoint of translation part of pose.
        m[:3] = (q[i, :3] + q[i + 1, :3]) / 2.

        # Extract angle from aa.
        a1 = np.linalg.norm(q[i, 3:])
        a2 = np.linalg.norm(q[i+1, 3:])

        # Midpoint of rotation axis.
        max = (q[i, 3:] / a1 + q[i+1, 3:] / a2) / 2.

        # Midpoint of angle
        mav1 = np.array([np.cos(a1), np.sin(a1)])
        mav2 = np.array([np.cos(a2), np.sin(a2)])
        mav = (mav1 + mav2)/2.
        ma = np.arctan2(mav[1], mav[0])

        m[3:] = max * ma

        ql.insert(i+1, m)
        q = np.array(ql)

    # Finally, rephase the angles.

    qa = np.linalg.norm(q[:, 3:], axis=1)
    q[:, 3:] /= qa[:, None]
    qa = np.unwrap(qa) + 2*np.pi
    q[:, 3:] *= qa[:, None]

    # TODO HACK just get rid of angle completely for now...
    q[:, 3:] = 1.


    return q
