import numpy as np
import scipy.linalg as sla


# DIRS
DATADIR = "data"
FIGDIR = "media"

# Click
ENVLIST = ["MRAC", "CMRAC", "FECMRAC", "RLCMRAC", "CLCMRAC"]

# Main system
A = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
B = np.array([[0], [1], [0]])
INITIAL_PARAM = np.vstack((-18.59521, 15.162375, -62.45153, 9.54708, 21.45291))
INITIAL_STATE = np.vstack((0.3, 0, 0))
AR = np.array([[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]])
BR = np.array([[0], [0], [-1]])
Q = np.eye(3)
P = sla.solve_lyapunov(A.T, -Q)

# Adaptive system
GAMMA = 1e4

# Command
COMMAND_LENGTH = 10
COMMAND_PHASE = 5
COMMAND_PATTERN = [1, 0, -1, 0]
