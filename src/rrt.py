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
                 samplegoal: float=.1, std:float =2.1):
        super().__init__()

        assert cov.ndim == 3
        assert mu.ndim == 2
        assert cov.shape[1] == mu.shape[1] == self.D/2
        assert cov.shape[0] == mu.shape[0]
        assert start.shape == goal.shape == (self.D,)
        assert len(bounds) == self.D

        self.Ainv = np.linalg.inv(np.linalg.cholesky(cov))
        self.mu = mu
        self.std = std
        self.bounds = bounds

        # Allocate the RRT 100k 6dof nodes.
        self._nodes = np.empty((100000, self.D))
        self.start = start
        self.goal = goal
        self.numnodes = 0
        ci = self._add(self.start)
        self.parents = {ci: None}

        # Pose > world for collision checking.
        xf = kn.th_6dof_rigid
        q = tt.dmatrix('q')
        self.f_xf = th.function(inputs=[q], outputs=xf(q, u), mode=th.compile.FAST_COMPILE)

        self.samplegoal = samplegoal

        self.done = False
        self._donepath = None

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
        xp = np.copy(self.nodes[pi])

        # Increment the step based on the sampled angle-axis dynamics, scale by distance.
        # TODO does this make sense with AA rep? hmm...
        x[3:] = xp[3:] + x[3:]*dd[pi]
        # TODO HACK Correct axis angle degeneracy.
        if np.linalg.norm(x[3:]) < .2:
            x[3:] = 1.

        # Go from 6dof to body point-cloud in 3d.
        uu = self.f_xf(np.vstack((x[None, :], xp[None, :])))
        # Check for obstacles along the path. If free, add to parent map.
        d = obs.np_el_nearestd(uu[0], uu[1], self.mu, self.Ainv)
        if not np.any(d < self.std):
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

        return np.flip(self.nodes[xx], axis=0)
