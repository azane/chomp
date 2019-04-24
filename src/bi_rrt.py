import numpy as np

# A simple, unoptimized RRT, specifically for the gm obstacle field.

class RRT_GM(object):
    def __init__(self, mu, cov):
        super().__init__()

        assert cov.ndim == 3
        assert mu.ndim == 2
        assert cov.shape[1] == self.mu.shape[1]
        assert cov.shape[0] == self.mu.shape[0]

        self.Ainv = np.linalg.inv(np.linalg.cholesky(cov))
        self.mu = mu
        self.d = self.mu.shape[1]

    # TODO move this logic to the obstacle checker and convert to a proper distance field...
    # TODO will need to put the spherical distance (in stdev units) on an axis, and reproject
    # TODO  back out to the covariant ellipse.
    def _backproject(self, x):
        # Project the points in x back to each gaussian in the field.
        assert x.shape[1] == self.d
        assert x.ndim == 2

        x = x[:, None, :] - self.mu[None, :, :]  # (X, N, D)
        x = np.matmul(self.Ainv, x[..., None])  # (N, D, D) . (X, N, D, 1)
        x = np.squeeze(x)  # (X, N, D)
        return x

    def check_collision(self, x1, x2, std=2.1):

        # First, backproject the points into each obstacle's spherical space.
        x1g = self._backproject(x1)  # (X, N, D)
        x2g = self._backproject(x2)

        # Compute the distance of closest approach between each pair of points in
        #  x1 and x2.
        # From http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

        # Our x0 is simply the origin here (because we projected back to a zero mean std=1 gaussian).

        x1gf = x1g.reshape(-1, self.d)
        x2gf = x2g.reshape(-1, self.d)

        num = np.cross(-x1gf, -x2gf)
        num = np.linalg.norm(num, axis=1)

        den = np.linalg.norm(x2gf - x1gf, axis=1)

        # Note: this distance will be the distance in std units.
        d = num / den
        d = d.reshape(x1g.shape[:-1])  # (X, N)

        # If within the std collision radius, there's a collision with this obstacle.
        c = d <= std
        # If a point pair collides with any obstacle, then we have a problem.
        c = np.any(c, axis=1)

        return c




