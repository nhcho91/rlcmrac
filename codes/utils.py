import numpy as np
from collections import deque
import pydash as ps


class Differentiator:
    def __init__(self):
        self.buffer = deque(maxlen=2)

    def append(self, t, x):
        self.buffer.append(np.hstack((t, x)))

    def get(self):
        buffer = self.buffer
        if len(buffer) == 0:
            raise ValueError("The buffer is empty.")
        if len(buffer) == 1:
            return np.zeros_like(buffer)[0, 1:]
        else:
            x = np.diff(buffer, axis=0)[0]
            return x[1:] / x[0]


def assign_2d(mat):
    mat = np.asarray(mat)
    if mat.ndim != 2:
        raise ValueError("The matrix should have 2 dimensions.")
    else:
        return mat


def set_attr(obj, attr_dict, keys=None, to_numpy=True):
    """Set attribute of an object from a dict"""
    attr_dict = ps.pick(attr_dict, keys or list(attr_dict.keys()))
    for attr, val in attr_dict.items():
        if isinstance(val, np.ndarray) or isinstance(val, list):
            val = np.asarray(val)
        setattr(obj, attr, val)
    return obj
