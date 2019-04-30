import theano as th
import theano.tensor as tt
import numpy as np

# From my own SO question/answer from a few years ago. Ha!
#  https://stackoverflow.com/questions/42169776/numpy-matmul-in-theano


def matmul(a: tt.TensorType, b: tt.TensorType, _left=False) -> tt.TensorVariable:
    """Replicates the functionality of numpy.matmul, except that
    the two tensors must have the same number of dimensions, and their ndim must exceed 1."""

    # TODO ensure that broadcastability is maintained if both a and b are broadcastable on a dim.

    assert a.ndim == b.ndim  # TODO support broadcasting for differing ndims.
    ndim = a.ndim
    assert ndim >= 2

    # If we should left multiply, just swap references.
    if _left:
        tmp = a
        a = b
        b = tmp

    # If a and b are 2 dimensional, compute their matrix product.
    if ndim == 2:
        return tt.dot(a, b)
    # If they are larger...
    else:
        # If a is broadcastable but b is not.
        if a.broadcastable[0] and not b.broadcastable[0]:
            # Scan b, but hold a steady.
            # Because b will be passed in as a, we need to left multiply to maintain
            #  matrix orientation.
            output, _ = th.scan(matmul, sequences=[b], non_sequences=[a[0], 1])
        # If b is broadcastable but a is not.
        elif b.broadcastable[0] and not a.broadcastable[0]:
            # Scan a, but hold b steady.
            output, _ = th.scan(matmul, sequences=[a], non_sequences=[b[0]])
        # If neither dimension is broadcastable or they both are.
        else:
            # Scan through the sequences, assuming the shape for this dimension is equal.
            output, _ = th.scan(matmul, sequences=[a, b])
        return output