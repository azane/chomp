import pytest
import src.objective as obj
import numpy as np


def test_priors_match():
    f, _, _ = obj.ffp_smoothness()
    for n in range(5, 100):
        q = np.random.rand(n, 3)

        naive = obj.slow_naive_prior(q)
        mat = obj.slow_fdmat_prior(q)
        ths = f(q)

        assert np.isclose(naive, mat)
        assert np.isclose(naive, ths)
