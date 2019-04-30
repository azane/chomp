import numpy as np
import src.kinematics as kn
import src.obstacle as obs
import theano as th
import theano.tensor as tt
from typing import *


# A simple, super unoptimized RRT, specifically for an elliptical obstacle field.
class RRT_GM6DOF(object):
    D = 6

    def __init__(self, mu, cov, u: tt.TensorConstant,
                 start: np.ndarray, goal: np.ndarray,
                 bounds: List[Tuple[float, float]],
                 samplegoal: float=.1,
                 dthresh: float=2.0):
        super().__init__()

        assert cov.ndim == 3
        assert mu.ndim == 2
        assert cov.shape[1] == mu.shape[1] == self.D/2
        assert cov.shape[0] == mu.shape[0]
        assert start.shape == goal.shape == (self.D,)
        assert len(bounds) == self.D
        assert dthresh > 0.

        self.Ainv = np.linalg.inv(np.linalg.cholesky(cov))
        self.mu = mu
        self.dthresh = dthresh
        self.bounds = bounds

        # Allocate the RRT 100k 6dof nodes.
        self._nodes = np.empty((100000, self.D))
        self.start = start
        self.goal = goal
        self.numnodes = 0
        ci = self._add(self.start)
        self.parents = {ci: None}

        # Pose > world for collision checking.
        # "Kinematics"
        xf = kn.th_6dof_rigid
        q = tt.dmatrix('q')
        tt_xf = xf(q, u)
        self.f_xf = th.function(inputs=[q], outputs=tt_xf, mode=th.compile.FAST_COMPILE)

        # Obstacle field checking.
        tt_mu = tt.constant(self.mu)
        tt_Ainv = tt.constant(self.Ainv)

        # DEBUB
        qtv = np.vstack((start[None, ...], (start-2.)[None, ...]))
        th.config.traceback.limit = 100
        # th.config.compute_test_value = 'warn'
        # q.tag.test_value = qtv
        # /DEBUG

        xf = tt.dtensor3('xf')
        tt_sdf = obs.th_el_nearestd_signed_wrap(mu=tt_mu, Ainv=tt_Ainv, dthresh=self.dthresh)(xf)
        # Clear if the sdf value is greater than 0.
        self.f_sdf = th.function(inputs=[xf], outputs=tt_sdf, mode=th.compile.FAST_COMPILE)

        # DEBUG
        x1_ = tt.dmatrix('x1_')
        x2_ = tt.dmatrix('x2_')
        d_ = obs.th_el_nearestd(x1=x1_, x2=x2_, mu=tt_mu, Ainv=tt_Ainv)
        self.f_d_ = th.function(inputs=[x1_, x2_], outputs=d_, mode=th.compile.FAST_COMPILE)
        # /DEBUG

        self.f_clr = lambda q_: np.all(self.f_sdf(self.f_xf(q_)) > 0.)

        self.samplegoal = samplegoal

        self.done = False
        self._donepath = None

    def f_clr2(self, q_):
        u = self.f_xf(q_)
        d = obs.np_el_nearestd(u[1:].reshape(-1, 3), u[:-1].reshape(-1, 3), self.mu, self.Ainv)
        return np.all(d > 0.)

    @property
    def nodes(self):
        return self._nodes[:self.numnodes]

    def _add(self, x: np.ndarray):
        self._nodes[self.numnodes] = x
        i = self.numnodes
        self.numnodes += 1
        return i

    def sample(self):
        if np.random.rand() < self.samplegoal:
            return self.goal
        else:
            # TODO Could cache a bunch of these sample for a pretty good speedup.
            x = []
            for b in self.bounds:
                x.append(np.random.uniform(b[0], b[1]))
            x = np.array(x)
            return x

    def plan(self):

        if np.allclose(self.nodes[-1], self.goal):
            self.done = True
            if self._donepath is None:
                self._donepath = self.path()
            return self._donepath

        if self.numnodes == len(self._nodes):
            raise MemoryError("Not enough nodes allocated!")

        # Sample a point in the cspace.
        x = self.sample()
        # Compute the distances to known nodes.
        # Compare only positions, we'll juse use dynamics for the angle-axis.
        dd = np.linalg.norm(self.nodes[:, :3] - x[None, :3], axis=1)
        # Get the index of the closest node. This is the proposed parent.
        pi = np.argmin(dd)
        # Retrieve the parent point.
        xp = self.nodes[pi]
        # Increment the step based on the sampled angle-axis dynamics, scale by distance.
        # TODO does this make sense with AA rep? hmm...
        xp[3:] += x[3:]*dd[pi]

        # Go from 6dof to body point-cloud in 3d.
        qq = np.vstack((x[None, :], xp[None, :]))
        # Check for obstacles along the path. If free, add to parent map.
        clr = self.f_clr2(qq)
        if clr:
            ci = self._add(x)
            self.parents[ci] = pi
            return self.path(ci)
        return None

    def path(self, ci=None):
        # Use the last seen node by default.
        if ci is None:
            ci = self.numnodes - 1

        xx = [ci]
        pi = self.parents[ci]
        while pi is not None:
            xx.append(pi)
            pi = self.parents[ci]
            ci = pi

        return self.nodes[xx]
