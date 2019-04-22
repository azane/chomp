import numpy as np
import theano as th
import theano.tensor as tt
import src.kinematics as kn


def test_unzero6dof():
    # Make sure that our unzeroing actually doesn't change anything.
    q = tt.dmatrix('q')
    q_ = np.random.rand(50, 6)

    th.config.compute_test_value = 'warn'
    q.tag.test_value = q_

    u = tt.constant(2.*(np.random.rand(100, 3) - .5))

    f_6dof = th.function(inputs=[q], outputs=kn.th_6dof_rigid(q, u))

    res1 = f_6dof(q_)
    res2 = f_6dof(kn.unzero_6dof(q_))

    assert np.allclose(res1, res2)