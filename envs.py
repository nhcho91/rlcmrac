import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from gym import spaces

from fym.core import BaseEnv, BaseSystem, infinite_box
import fym.logging as logging

from utils import assign_2d


class Mrac(BaseEnv):
    def __init__(self, spec, data_callback=None):
        A = Ar = spec['reference_system']['Ar']
        B = spec['main_system']['B']
        Br = spec['reference_system']['Br']
        self.unc = ParamUnc(initial_param=spec['main_system']['initial_param'])
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

    def step(self, action):
        states = self.states
        time = self.clock.get()

        next_states, full_hist = self.get_next_states(time, states, action)

        # Reward
        reward = 0

        # Terminal condition
        done = self.is_terminal()

        # Info
        info = {
            "time": time,
            "state": states,
            "control": self.get_control(states),
            "cmd": self.cmd.get(time),
            "reward": reward,
            "real_param": self.unc.get_param(time)
        }

        # Updates
        self.states = next_states
        self.clock.tick()

        return states, reward, done, info

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W = states.values()

        e = x - xr
        u = self.get_control(states)

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

    def close(self):
        super().close()
        if self.data_callback is not None:
            data = logging.load(self.logger.path)
            self.data_callback(self, data)

    def get_control(self, states):
        x, W = states["main_system"], states["adaptive_system"]
        u = -W.T.dot(self.unc.basis(x))
        return u


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

    def wrap_action(self, memory, action):
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


class ParamUnc:
    def __init__(self, initial_param):
        self.W = assign_2d(initial_param)

    def get(self, t, x):
        return self.get_param(t).T.dot(self.basis(x))

    def get_param(self, t):
        return self.W + 2 * np.sin(t) + 30 * np.tanh(t / 60)

    def basis(self, x):
        return np.hstack((x[:2], np.abs(x[:2]) * x[1], x[0]**3))


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


class MemorySystem(BaseSystem):
    def __init__(self, initial_state):
        super().__init__(initial_state=initial_state)

    def deriv(self, x, k, u1, u2):
        xdot = - k * x + np.outer(u1, u2)
        return xdot


class MainSystem(BaseSystem):
    def __init__(self, initial_state, A, B, Br, unc, cmd):
        super().__init__(initial_state=initial_state)
        self.A, self.B, self.Br = map(assign_2d, [A, B, Br])
        self.unc = unc
        self.cmd = cmd

    def deriv(self, t, x, u):
        xdot = (
            self.A.dot(x)
            + self.B.dot(u + self.unc.get(t, x))
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


def fill_nan(arr, size, axis=0):
    if arr.shape[axis] < size:
        pad_width = [[0, 0] for _ in range(np.ndim(arr))]
        pad_width[axis][1] = size - arr.shape[axis]
        return np.pad(arr, pad_width, constant_values=np.nan)
    else:
        return arr
