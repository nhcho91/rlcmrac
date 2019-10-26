from collections import OrderedDict
from itertools import count

import pydash as ps

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.signal

from gym import spaces, Wrapper

from fym.core import BaseEnv, BaseSystem, infinite_box
import fym.logging as logging

from utils import assign_2d


class MracEnv(BaseEnv):
    def __init__(self, spec, data_callback=None):
        A = Ar = spec['reference_system']['Ar']
        B = spec['main_system']['B']
        Br = spec['reference_system']['Br']
        self.unc = ParamUnc(real_param=spec['main_system']['real_param'])
        self.cmd = SquareCmd(
            length=spec["command"]["length"],
            phase=spec["command"]["phase"],
            pattern=spec["command"]["pattern"]
        )
        self.data_callback = data_callback

        systems = {
            'main_system': MainSystem(
                initial_state=spec['main_system']['initial_state'],
                A=A,
                B=B,
                Br=Br,
                unc=self.unc,
                cmd=self.cmd,
            ),
            'reference_system': RefSystem(
                initial_state=spec['reference_system']['initial_state'],
                Ar=Ar,
                Br=Br,
                cmd=self.cmd
            ),
            'adaptive_system': AdaptiveSystem(
                initial_state=spec['adaptive_system']['initial_state'],
                A=A,
                B=B,
                gamma1=spec['adaptive_system']['gamma1'],
                Q=spec['adaptive_system']['Q'],
                unc=self.unc
            )
        }

        self.observation_space = infinite_box((
            len(spec['main_system']['initial_state'])
            + len(spec['reference_system']['initial_state']),
            + len(np.ravel(spec['adaptive_system']['initial_state'])),
            + np.shape(Br)[1],
        ))
        self.action_space = infinite_box([])

        super().__init__(
            systems=systems,
            dt=spec['environment']['time_step'],
            max_t=spec['environment']['final_time'],
            ode_step_len=spec['environment']['ode_step_len'],
        )

    def reset(self):
        """
        Resets to the initial states of the systems.
        """
        states = super().reset()
        t = self.clock.get()
        return self.observation(t, states)

    def observation(self, t, states):
        """
        This function converts all the states to a flat array.
        """
        x, xr, W, = states.values()
        obs = np.hstack((
            x,
            xr,
            W.ravel(),
            self.cmd.get(t),
        ))
        return obs

    def step(self, action):
        states = self.states
        t = self.clock.get()

        next_states, full_hist = self.get_next_states(t, states, action)

        # Reward
        reward = self.compute_reward()

        # Terminal condition
        done = self.is_terminal()

        # Updates
        self.states = next_states
        self.clock.tick()

        t = self.clock.get()
        obs = self.observation(t, next_states)
        return obs, reward, done, {}

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W = states.values()

        e = x - xr
        u = -W.T.dot(self.unc.basis(x))

        xdot = {
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, xr),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e),
        }
        return self.unpack_state(xdot)

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False

    def compute_reward(self):
        return None

    def parse_data(self, path):
        data = logging.load(path)
        xs = data['state']['main_system']
        Ws = data['state']['adaptive_system']

        cmd = np.hstack([self.cmd.get(t) for t in data['time']])
        u_mrac = np.vstack(
            [-W.T.dot(self.unc.basis(x)) for W, x in zip(Ws, xs)])

        data.update({
            "control": {
                "MRAC": u_mrac,
            },
            "cmd": cmd,
        })
        return data

    def close(self):
        super().close()
        if self.data_callback is not None:
            data = logging.load(self.logger.path)
            self.data_callback(self, data)


class CmracEnv(MracEnv):
    def __init__(self, spec, data_callback):
        super().__init__(spec, data_callback)
        self.gamma2 = spec['composite_system']['gamma2']
        self.mem_max_size = spec["composite_system"]["memory"]["max_size"]
        self.eps = spec['composite_system']['memory']['norm_eps']

        new_systems = {
            'filter_system': FilterSystem(
                initial_state=spec['filter_system']['initial_state'],
                A=spec['reference_system']['Ar'],
                B=spec['main_system']['B'],
                tau=spec['filter_system']['tau']
            ),
            'filtered_phi': FilteredPhi(
                initial_state=spec['filtered_phi']['initial_state'],
                tau=spec['filter_system']['tau'],
                unc=self.unc
            ),
        }
        self.append_systems(new_systems)

        W_shape = np.shape(spec["adaptive_system"]["initial_state"])
        self.observation_space = infinite_box((
            len(spec['main_system']['initial_state'])
            + len(spec['reference_system']['initial_state'])
            + len(np.ravel(spec['adaptive_system']['initial_state']))
            + np.shape(spec["reference_system"]["Br"])[1]
            + self.mem_max_size * np.sum(W_shape),
        ))
        self.action_space = infinite_box((self.mem_max_size,))

        # Memory
        self.phi_size = W_shape[0]
        self.y_size = W_shape[1]
        self.memory = {}

    def reset(self):
        states = super(MracEnv, self).reset()
        time = self.clock.get()
        memory = {
            't': np.empty((0,)),
            'phi': np.empty((0, self.phi_size)),
            'y': np.empty((0, self.y_size))
        }
        self.memory = self.update_memory(memory, time, states)
        return self.observation(time, states, self.memory)

    def update_memory(self, memory, time, states):
        """
        This function stacks the state informations to the memories,
        and returns them.
        """
        x, xr, _, z, phif = states.values()
        e = x - xr
        y = self.systems['filter_system'].get_y(z, e)
        norm_phif = nla.norm(phif) + self.eps

        t_mem = np.hstack((memory['t'], time))
        phi_mem = np.vstack((memory['phi'], phif / norm_phif))
        y_mem = np.vstack((memory['y'], y / norm_phif))
        memory = {'t': t_mem, 'phi': phi_mem, 'y': y_mem}
        return memory

    def observation(self, time, states, memory):
        """
        This function converts all the states including the memories to a
        flat array.
        """
        x, xr, W, _, _ = states.values()
        obs = np.hstack((
            x,
            xr,
            W.ravel(),
            self.cmd.get(time),
            np.hstack(memory['phi']),
            np.hstack(memory['y'])
        ))
        return obs

    def step(self, action):
        states = self.states
        memory = self.memory
        t = self.clock.get()

        wrapped_action = self.wrap_action(memory, action)
        next_states, full_hist = self.get_next_states(t, states, wrapped_action)

        reduced_memory = self.reduce_memory(memory, action)

        # Reward
        reward = self.compute_reward(states, wrapped_action)

        # Terminal condition
        done = self.is_terminal()

        # Info
        info = {
            "time": t,
            "states": states,
            "action": action,
            "memory": {
                "time": memory["t"]
            }
        }

        # Updates
        self.states = next_states
        self.clock.tick()

        t = self.clock.get()
        next_memory = self.update_memory(reduced_memory, t, next_states)
        self.memory = next_memory

        obs = self.observation(t, next_states, next_memory)
        return obs, reward, done, info

    def wrap_action(self, memory, action):
        phi_mem, y_mem = memory["phi"], memory["y"]
        M = mul_dist(action, phi_mem, phi_mem)
        N = mul_dist(action, phi_mem, y_mem)
        action = {'M': M, 'N': N}
        return action

    def reduce_memory(self, memory, action):
        t_mem, phi_mem, y_mem = memory["t"], memory["phi"], memory["y"]
        if len(phi_mem) == self.mem_max_size:
            argmin = np.argmin(action)
            t_mem = np.delete(t_mem, argmin, axis=0)
            phi_mem = np.delete(phi_mem, argmin, axis=0)
            y_mem = np.delete(y_mem, argmin, axis=0)

        memory = {'t': t_mem, 'phi': phi_mem, 'y': y_mem}
        return memory

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W, z, phif = states.values()
        M, N = action['M'], action['N']

        e = x - xr
        u = -W.T.dot(self.unc.basis(x))

        xdot = OrderedDict.fromkeys(self.systems)
        xdot.update({
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, xr),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e) - np.dot(self.gamma2, M.dot(W) - N),
            'filter_system': self.systems['filter_system'].deriv(z, e, u),
            'filtered_phi': self.systems['filtered_phi'].deriv(phif, x),
        })
        return self.unpack_state(xdot)

    def compute_reward(self, states, action):
        x, xr, _, _, _ = states.values()
        e = x - xr
        min_eigval = nla.eigvals(action["M"]).min()
        e_cost = e.dot(e)
        reward = 1e2*min_eigval - 1e-2*e_cost
        return reward


class ParamUnc:
    def __init__(self, real_param):
        self.W = assign_2d(real_param)

    def get(self, state):
        return self.W.T.dot(self.basis(state))

    def basis(self, x):
        return np.hstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))


class SquareCmd:
    def __init__(self, length, phase=0, pattern=[1, -1]):
        """
        Parameters
        ----------
        period: float (>0) [sec]
        amplitude: float (>0)
        width: float or int in (0, 100) [%]
        phase: float [sec]
        """
        self.length = length
        self.phase = phase
        self.pattern = pattern
        self.period = self.length * len(self.pattern)

    def get(self, t):
        """
        scipy.signal.square:
            The square wave has a period ``2*pi``, has value +1 from 0 to
            ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``.
            `duty` must be in the interval [0,1].
        """
        t = t - self.phase
        if t > 0:
            s = self.pattern[
                int((t % self.period) / self.period * len(self.pattern))
            ]
        else:
            s = 0
        return np.atleast_1d(s)


class MainSystem(BaseSystem):
    def __init__(self, initial_state, A, B, Br, unc, cmd):
        super().__init__(initial_state=initial_state)
        self.A, self.B, self.Br = map(assign_2d, [A, B, Br])
        self.unc = unc
        self.cmd = cmd

    def deriv(self, t, x, u):
        xdot = (
            self.A.dot(x)
            + self.B.dot(u + self.unc.get(x))
            + self.Br.dot(self.cmd.get(t))
        )
        return xdot


class RefSystem(BaseSystem):
    def __init__(self, initial_state, Ar, Br, cmd):
        super().__init__(initial_state)

        self.Ar = assign_2d(Ar)
        self.Br = assign_2d(Br)
        self.cmd = cmd

    def deriv(self, t, x):
        xdot = self.Ar.dot(x) + self.Br.dot(self.cmd.get(t))
        return xdot


class AdaptiveSystem(BaseSystem):
    def __init__(self, initial_state, A, B, gamma1, Q, unc):
        super().__init__(initial_state=initial_state)

        self.A = assign_2d(A)
        self.B = assign_2d(B)
        self.gamma1 = gamma1
        self.Q = assign_2d(Q)
        self.P = self.calc_P(self.A, self.B, self.Q)
        self.basis = unc.basis

    def calc_P(self, A, B, Q):
        P = sla.solve_lyapunov(self.A.T, -self.Q)
        return P

    def deriv(self, W, x, e):
        # M, N = composite_input['M'], composite_input['N']
        Wdot = (
            np.dot(
                self.gamma1, np.outer(self.basis(x), e)
            ).dot(self.P).dot(self.B)
        )
        return Wdot


class FilterSystem(BaseSystem):
    def __init__(self, initial_state, A, B, tau):
        super().__init__(initial_state=initial_state)
        self.A = A
        self.Bh = sla.pinv(B)
        self.tau = tau

    def deriv(self, z, e, u):
        zdot = (
            1 / self.tau * (self.Bh.dot(e) - z)
            + self.Bh.dot(self.A).dot(e)
            + u
        )
        return zdot

    def get_y(self, z, e):
        y = 1 / self.tau * (self.Bh.dot(e) - z)
        return y


class FilteredPhi(BaseSystem):
    def __init__(self, initial_state, tau, unc):
        super().__init__(initial_state=initial_state)
        self.tau = tau
        self.basis = unc.basis

    def deriv(self, phif, x):
        phif_dot = -1 / self.tau * (phif - self.basis(x))
        return phif_dot


def mul_dist(dist, phi_mem, xs):
    res = sum([
        d * np.outer(phi, x) for d, phi, x in zip(dist, phi_mem, xs)
    ])
    return res
