import theano as th
import theano.tensor as tt
import numpy as np
from src.th_matmul import matmul

# From my own SO question/answer from a few years ago. Ha!
#  https://stackoverflow.com/questions/42169776/numpy-matmul-in-theano


def test_th_matmul():
    vlist = []
    flist = []
    ndlist = []
    for i in range(2, 30):
        dims = int(np.random.random() * 4 + 2)

        # Create a tuple of tensors with potentially different broadcastability.
        vs = tuple(
            tt.TensorVariable(
                tt.TensorType('float64',
                              tuple((p < .3) for p in np.random.ranf(dims-2))
                              # Make full matrices
                              + (False, False)
                )
            )
            for _ in range(2)
        )
        vs = tuple(tt.swapaxes(v, -2, -1) if j % 2 == 0 else v for j, v in enumerate(vs))

        f = th.function([*vs], [matmul(*vs)])

        # Create the default shape for the test ndarrays
        defshape = tuple(int(np.random.random() * 5 + 1) for _ in range(dims))
        # Create a test array matching the broadcastability of each v, for each v.
        nds = tuple(
            np.random.ranf(
                tuple(s if not v.broadcastable[j] else 1 for j, s in enumerate(defshape))
            )
            for v in vs
        )
        nds = tuple(np.swapaxes(nd, -2, -1) if j % 2 == 0 else nd for j, nd in enumerate(nds))

        ndlist.append(nds)
        vlist.append(vs)
        flist.append(f)

    for i in range(len(ndlist)):
        assert np.allclose(flist[i](*ndlist[i]), np.matmul(*ndlist[i]))


