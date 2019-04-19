import pytest
import src.objective_slow as obj_slow
import numpy as np


def test_priors_match():
    for n in range(5, 100):
        q = np.random.rand(n, 3)

        naive = obj_slow.slow_naive_prior(q)
        mat = obj_slow.slow_fdmat_prior(q)

        assert np.isclose(naive, mat)
