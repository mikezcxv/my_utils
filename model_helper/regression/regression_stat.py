import numpy as np


# def rmsle_vector(y, y0):
#     assert len(y) == len(y0)
#     # assert prediction > 0
#     y0[y0 <= 0] = 0.000000001
#     return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2))) * -1


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))
