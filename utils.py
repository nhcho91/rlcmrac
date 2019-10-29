import numpy as np
from collections import deque
import ujson


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
    if not (mat.ndim == 2 or mat.ndim == 0):
        raise ValueError("The matrix should have 2 dimensions.")
    else:
        return mat


def load_spec(file):
    with open(file, 'r') as f:
        data = ujson.load(f)
    return data
