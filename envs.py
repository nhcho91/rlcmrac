import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

import tqdm
from gym import spaces

from fym.core import BaseEnv, BaseSystem, infinite_box
import fym.logging as logging

import config
from utils import assign_2d


def basis(x):
    return np.vstack((x[:2], np.abs(x[:2]) * x[1], x[0]**3))


class ParamUnc:
    def __init__(self, initial_param, time_varying=False):
        self.W = assign_2d(initial_param)
        self.time_varying = time_varying

    def get(self, t, x):
        return self.get_param(t).T.dot(basis(x))

    def get_param(self, t):
        if self.time_varying:
            return self.W + 20 * np.tanh(t / 60) + 30 * np.sin(t / 20)
        else:
            return self.W


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
        return np.atleast_2d(s)


class MainSystem(BaseSystem):
    A = config.A
    B = config.B
    Br = config.BR

    def set_dot(self, u, unc, cmd):
        self.dot = self.A.dot(self.state) + self.B.dot(u + unc) + self.Br.dot(cmd)


class RefSystem(BaseSystem):
    Ar = config.AR
    Br = config.BR

    def set_dot(self, c):
        self.dot = self.Ar.dot(self.state) + self.Br.dot(c)


class Mrac(BaseEnv):
    def __init__(self, gamma):
        super().__init__()
        self.xr = RefSystem(config.INITIAL_STATE)
        self.W = BaseSystem(np.zeros_like(config.INITIAL_PARAM))
        self.Gamma = np.asarray(gamma)

    def set_dot(self, x, c):
        e = x - self.xr.state
        self.xr.set_dot(c)
        self.W.dot = self.Gamma.dot(basis(x)).dot(e.T).dot(config.P).dot(config.B)

    def get(self, x, W):
        return -W.T.dot(basis(x))


class Sim1(BaseEnv):
    def __init__(self, gamma=config.GAMMA, **kwargs):
        super().__init__(**kwargs)
        self.main = MainSystem(config.INITIAL_STATE)
        self.mrac = Mrac(gamma)

        self.unc = ParamUnc(initial_param=config.INITIAL_PARAM)
        self.cmd = SquareCmd(
            length=config.COMMAND_LENGTH,
            phase=config.COMMAND_PHASE,
            pattern=config.COMMAND_PATTERN
        )

    def set_dot(self, t):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, (_, W) = self.observe_list()

        u = self.mrac.get(x, W)

        unc = self.unc.get(t, x)
        cmd = self.cmd.get(t)

        self.main.set_dot(u, unc, cmd)
        self.mrac.set_dot(x, cmd)

    def step(self):
        states = self.observe_dict()
        control = self.mrac.get(states["main"], states["mrac"]["W"])
        time = self.clock.get()
        cmd = self.cmd.get(time)
        param = self.unc.get_param(time)
        *_, done = self.update()

        # Info
        info = {
            "time": time,
            "state": states,
            "control": control,
            "cmd": cmd,
            "real_param": param,
        }

        return done, info


class Cmrac(Mrac):
    def __init__(self, spec, data_callback=None):
        super().__init__(spec, data_callback)
        self.gamma2 = spec['composite_system']['gamma2']
        self.norm_eps = spec["memory"]["norm_eps"]

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

        self.M_shape = self.systems["adaptive_system"].state_shape[:1] * 2
        self.N_shape = self.systems["adaptive_system"].state_shape

        self.action_space = spaces.Dict({
            "M": infinite_box(self.M_shape),
            "N": infinite_box(self.N_shape)
        })

    def step(self, action):
        """
        Parameters
        ----------
        action: dict
            A dictionary with keys `M` and `N`.
        """
        next_states, reward, done, info = super().step(action)

        eigvals = nla.eigvals(action["M"])

        info.update({
            "eigs": eigvals,
            "M": action["M"],
            "N": action["N"],
        })
        return next_states, reward, done, info

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W, z, phif = states.values()
        M, N = action['M'], action['N']

        e = x - xr
        u = self.get_control(states)

        xdot = {
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, xr),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e) - np.dot(self.gamma2, M.dot(W) - N),
            'filter_system': self.systems['filter_system'].deriv(z, e, u),
            'filtered_phi': self.systems['filtered_phi'].deriv(phif, x),
        }
        return self.unpack_state(xdot)


class FeCmrac(Cmrac):
    def __init__(self, spec, data_callback=None):
        super().__init__(spec, data_callback)
        self.kl = spec["fecmrac"]["kl"]
        self.ku = spec["fecmrac"]["ku"]
        self.theta = spec["fecmrac"]["theta"]

        new_systems = {
            'omega_system': MemorySystem(
                initial_state=np.zeros(self.M_shape),
            ),
            'm_system': MemorySystem(
                initial_state=np.zeros(self.N_shape),
            ),
        }
        self.append_systems(new_systems)
        self.action_space = infinite_box((0,))

    def reset(self):
        states = super().reset()
        omega_a, m_a = states['omega_system'], states['m_system']
        memory = {
            'time': self.clock.get(),
            'saved_eig': nla.eigvals(omega_a).min(),
            'omega_a': omega_a,
            'm_a': m_a
        }
        self.memory = memory
        return states

    def step(self, action):
        states = self.states
        memory = self.memory
        action = {"M": memory["omega_a"], "N": memory["m_a"]}

        next_states, reward, done, info = super().step(action)

        # Info
        k = self.get_k(states)
        eigvals = nla.eigvals(states["omega_system"])

        info.update({
            "memory": memory,
            "k": k,
            "hidden_eigs": eigvals
        })

        # Update the buffer
        next_time = self.clock.get()

        next_omega, next_m = [
            next_states[k] for k in ['omega_system', 'm_system']
        ]
        next_eig = nla.eigvals(next_omega).min()
        if next_eig >= memory['saved_eig']:
            next_memory = {
                **memory,
                'time': next_time,
                'saved_eig': next_eig,
                'omega_a': next_omega,
                'm_a': next_m
            }
        else:
            next_memory = memory

        self.memory = next_memory
        return next_states, reward, done, info

    def get_k(self, states):
        x, phif = states['main_system'], states['filtered_phi']
        phif_norm = nla.norm(phif) + self.norm_eps
        dot_phif = self.systems['filtered_phi'].deriv(phif, x)
        normed_dot_phif = dot_phif / phif_norm
        k = (
            self.kl
            + (self.ku - self.kl) * np.tanh(
                self.theta * nla.norm(normed_dot_phif)
            )
        )
        return k

    def derivs(self, t, states, action):
        x, xr, W, z, phif, omega, m = states.values()
        M, N = action["M"], action["N"]

        e = x - xr
        u = self.get_control(states)

        phif_norm = nla.norm(phif) + self.norm_eps
        normed_y = self.systems["filter_system"].get_y(z, e) / phif_norm
        normed_phif = phif / phif_norm
        k = self.get_k(states)

        xdot = {
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, xr),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e) - np.dot(self.gamma2, M.dot(W) - N),
            'filter_system': self.systems['filter_system'].deriv(z, e, u),
            'filtered_phi': self.systems['filtered_phi'].deriv(phif, x),
            'omega_system': self.systems['omega_system'].deriv(
                omega, k, normed_phif, normed_phif),
            'm_system': self.systems['m_system'].deriv(
                m, k, normed_phif, normed_y),
        }
        return self.unpack_state(xdot)


class RlCmrac(Cmrac):
    def __init__(self, spec, data_callback=None):
        super().__init__(spec, data_callback)
        self.mem_max_size = spec["memory"]["max_size"]
        self.norm_eps = spec["memory"]["norm_eps"]

        W_shape = np.shape(spec["adaptive_system"]["initial_state"])
        self.phi_size = W_shape[0]
        self.y_size = W_shape[1]

        self.observation_space = infinite_box((
            len(spec['main_system']['initial_state'])
            + len(spec['reference_system']['initial_state'])
            + len(np.ravel(spec['adaptive_system']['initial_state']))
            + np.shape(spec["reference_system"]["Br"])[1]
            + self.mem_max_size * np.sum(W_shape),
        ))
        self.action_space = infinite_box((self.mem_max_size,))

    def reset(self):
        states = super().reset()
        time = self.clock.get()
        memory = {
            't': np.empty((0,)),
            'phi': np.empty((0, self.phi_size)),
            'y': np.empty((0, self.y_size))
        }
        memory = self.update_memory(memory, time, states)

        self.states = states
        self.memory = memory
        return self.observation(time, states, memory)

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

    def update_memory(self, memory, time, states):
        """
        This function stacks the state informations to the memories,
        and returns them.
        """
        x, xr, _, z, phif = states.values()
        e = x - xr
        y = self.systems['filter_system'].get_y(z, e)
        phif_norm = nla.norm(phif) + self.norm_eps

        t_mem = np.hstack((memory['t'], time))
        phi_mem = np.vstack((memory['phi'], phif / phif_norm))
        y_mem = np.vstack((memory['y'], y / phif_norm))
        memory = {'t': t_mem, 'phi': phi_mem, 'y': y_mem}
        return memory

    def step(self, action):
        states = self.states
        memory = self.memory

        wrapped_action = self.wrap_action(memory, action)
        next_states, reward, done, info = super().step(wrapped_action)

        reduced_memory, removed_t = self.reduce_memory(memory, action)

        # Reward
        x, xr, _, _, _ = states.values()
        e = x - xr
        eig_cost = info["eigs"].min()
        e_cost = e.dot(e)
        u = info['control']
        u_cost = u.dot(u)
        reward = 1e5 * eig_cost - 1e2 * e_cost - 1e-1 * u_cost

        # Terminal condition
        done = self.is_terminal()

        # Info
        info_memory = {k: fill_nan(v, self.mem_max_size)
                       for k, v in memory.items()}

        info.update({
            "dist": action,
            "M": wrapped_action["M"],
            "N": wrapped_action["N"],
            "memory": info_memory,
            "removed": removed_t,
            "tracking_error": e_cost,
        })

        # Update
        next_time = self.clock.get()
        next_memory = self.update_memory(reduced_memory, next_time, next_states)
        self.memory = next_memory

        return (
            self.observation(next_time, next_states, next_memory),
            reward,
            done,
            info
        )

    def wrap_action(self, memory, action=None):
        if action is None:
            action = [1 / self.mem_max_size] * len(memory["t"])

        phi_mem, y_mem = memory["phi"], memory["y"]
        M = mul_dist(action, phi_mem, phi_mem)
        N = mul_dist(action, phi_mem, y_mem)
        action = {'M': M, 'N': N}
        return action

    def reduce_memory(self, memory, action):
        t_mem, phi_mem, y_mem = memory["t"], memory["phi"], memory["y"]
        removed_t = np.nan
        if len(phi_mem) == self.mem_max_size:
            argmin = np.argmin(action)
            removed_t = t_mem[argmin]
            t_mem = np.delete(t_mem, argmin, axis=0)
            phi_mem = np.delete(phi_mem, argmin, axis=0)
            y_mem = np.delete(y_mem, argmin, axis=0)

        memory = {'t': t_mem, 'phi': phi_mem, 'y': y_mem}
        return memory, removed_t


class ClCmrac(RlCmrac):
    def step(self, action):
        states = self.states
        memory = self.update_memory(
            self.memory, self.clock.get(), states
        )

        wrapped_action = self.wrap_action(memory)
        next_states, reward, done, info = (
            super(RlCmrac, self).step(wrapped_action)
        )

        # Info
        info_memory = {k: fill_nan(v, self.mem_max_size)
                       for k, v in memory.items()}

        info.update({
            "M": wrapped_action["M"],
            "N": wrapped_action["N"],
            "memory": info_memory,
        })

        # Update
        self.memory = memory

        return next_states, reward, done, info

    def update_memory(self, memory, time, states):
        best_memory = memory

        current_len = len(memory["t"])
        if self.mem_max_size == current_len:
            M = self.wrap_action(best_memory)["M"]
            best_eig = nla.eigvals(M).min()

            for i in range(current_len):
                reduced_memory = {}
                for key, val in memory.items():
                    reduced_memory[key] = np.delete(val, i, axis=0)

                tmp_memory = super().update_memory(reduced_memory, time, states)
                M = self.wrap_action(tmp_memory)["M"]
                min_eig = nla.eigvals(M).min()

                if min_eig > best_eig:
                    best_memory = tmp_memory
                    best_eig = min_eig

        else:
            best_memory = super().update_memory(best_memory, time, states)

        return best_memory


class MemorySystem(BaseSystem):
    def __init__(self, initial_state):
        super().__init__(initial_state=initial_state)

    def deriv(self, x, k, u1, u2):
        xdot = - k * x + np.outer(u1, u2)
        return xdot


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


def fill_nan(arr, size, axis=0):
    if arr.shape[axis] < size:
        pad_width = [[0, 0] for _ in range(np.ndim(arr))]
        pad_width[axis][1] = size - arr.shape[axis]
        return np.pad(arr, pad_width, constant_values=np.nan)
    else:
        return arr
