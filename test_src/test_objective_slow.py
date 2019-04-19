import pytest
import src.objective_slow as obj_slow
import numpy as np


def test_priors_match():
    for n in range(5, 100):
        q = np.random.rand(n, 3)

        dt = 1.
        naive = obj_slow.slow_naive_prior(q, dt)
        mat = obj_slow.slow_fdmat_prior(q, dt)
        quad = obj_slow.slow_quad_prior(q, dt)

        assert np.isclose(naive, mat)
        assert np.isclose(naive, quad)