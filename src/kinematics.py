import numpy as np
from theano.ifelse import ifelse
import theano.tensor as tt
from .th_matmul import matmul


def th_translation_only(q: tt.TensorVariable, u: tt.TensorConstant):
    assert q.ndim == 2
    assert u.ndim == 2

    # Simple "kinematics" function that just translates the body elements
    #  according to q. q is in world frame, u is in robot frame.

    qu = q[:, None, :] + u[None, :, :]  # .shape == (Q, U, D)

    return qu


def th_6dof_translation_only(q: tt.TensorVariable, u: tt.TensorConstant):
    assert q.ndim == 2
    assert u.ndim == 2

    qu = q[:, None, :3] + u[None, :, :]  # .shape == (Q, U, D)

    return qu


# <Theano 6DOF Kinematics>
# From https://en.wikipedia.org/wiki/Rotation_matrix (axis and angle conversion to rotation matrix).

# For broadcastability, I'm using these sparse matrices that we can multiply by things.
# I believe theano will optimize out the multiplications by zero.
s_ = tt.constant(np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
]))

c_ = tt.constant(np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]))

x_ = tt.constant(np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
]))

y_ = tt.constant(np.array([
    [0, 0, 1],
    [0, 0, 0],
    [1, 0, 0],
]))

z_ = tt.constant(np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]))


def th_6dof_rigid(q: tt.TensorVariable, u: tt.TensorConstant):

    # q  # .shape == (Q, 6)
    tr = q[:, :3]  # .shape == (Q, 3)  # translation
    aa = q[:, 3:]  # .shape == (Q, 3)  # angle axis
    # u  # .shape == (U, 3)

    # Extract theta from scaled axis angle.
    # NOTE: We assume issues with zeroed controls are taken care of outside. "Fixing" this here
    #  would result in discontinuities in the gradients (at least, using methods I'm aware of would).
    # (Also: There's a discontinuity at zero here anyway...)
    # See unzero_6dof below for a function to do this globally (which is quite helpful b/c of CHOMP's
    #  globalization of the smoothness)
    theta = aa.norm(2, axis=1)  # .shape == (Q)  # angle
    ax = aa / theta.dimshuffle(0, 'x')  # .shape == (Q, 3)  # axis

    # .shape == (Q)
    c = tt.cos(theta)
    s = tt.sin(theta)
    C = 1. - c

    # Prep the easily-broadcasted part of the rotation matrix.
    R_ = (ax[:, :, None] * ax[:, None, :]) * C[:, None, None]  # .shape == (Q, 3, 3)

    # .shape == (Q, 3, 3)
    ss = s_[None, :, :] * s[:, None, None]
    cc = c_[None, :, :] * c[:, None, None]
    # .shape == (Q, 3, 3)  # the index doesn't preserve an axis.
    xx = x_[None, :, :] * ax[:, 0, None, None]
    yy = y_[None, :, :] * ax[:, 1, None, None]
    zz = z_[None, :, :] * ax[:, 2, None, None]

    # Combine everything.
    R = R_ + cc + ss * (xx + yy + zz)  # .shape == (Q, 3, 3)

    # Apply the rotations to u and translate.
    # (Q, 1, 3, 3) . (1, U, 3, 1) == (Q, U, 3, 1)
    urot = tt.squeeze(matmul(R.dimshuffle(0, 'x', 1, 2), u.dimshuffle('x', 0, 1, 'x')))
    # (Q, 1, 3) + (Q, U, 3) = (Q, U, 3)
    ufin = tr.dimshuffle(0, 'x', 1) + urot

    return ufin


def unzero_6dof(q: np.ndarray):
    # TODO actually, we could roll this into the above pretty easily...

    # NOTE: We do this globally b/c CHOMP maintains global smoothness, and I don't want to deal
    #  with making my gradients polar or whatever. ; )

    # Pull the angles and axes out.
    aa = q[:, 3:]
    theta = np.linalg.norm(aa, axis=1)[:, None]
    ax = aa / theta

    # Add 2pi to the angles. This changes nothing.
    theta += 2. * np.pi

    # Rescale the axes and return.
    return np.hstack((q[:, :3], ax * theta))

# </Theano 6DOF Kinematics>
