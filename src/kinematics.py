import numpy as np
import theano as th
import theano.tensor as tt


def th_translation_only(q: tt.TensorVariable, u: tt.TensorConstant):
    assert q.ndim == 2
    assert u.ndim == 2

    # Simple "kinematics" function that just translates the body elements
    #  according to q. q is in world frame, u is in robot frame.

    qu = q[:, None, :] + u[None, :, :]  # .shape == (Q, U, D)

    return qu


def th_6dof_rigid(q: tt.TensorVariable, u: tt.TensorConstant):
    raise NotImplementedError()