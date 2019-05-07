import theano.tensor as tt
import numpy as np

U = 20
D = 3


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