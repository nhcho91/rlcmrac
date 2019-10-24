from collections import OrderedDict
from itertools import count

import pydash as ps

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.signal

from gym import spaces, Wrapper

from fym.core import BaseEnv, BaseSystem, infinite_box
import fym.utils.logger as logger

from utils import assign_2d


class MRACEnv(BaseEnv):
    def __init__(self, spec):
        A = Ar = spec['reference_system']['Ar']
        B = spec['main_system']['B']
        Br = spec['reference_system']['Br']
        self.eps = spec['agent']['eps']

        self.obs_logger = logger.Logger(
            file_name=spec['environment']['obs_log_name']
        )

        self.unc = ParamUnc(real_param=spec['main_system']['real_param'])
        self.cmd = SquareCmd(period=20, phase=10)

        main_system = MainSystem(
            name='main_system',
            initial_state=spec['main_system']['initial_state'],
            A=A,
            B=B,
            Br=Br,
            unc=self.unc,
            cmd=self.cmd
        )
        reference_system = RefSystem(
            name='reference_system',
            initial_state=spec['reference_system']['initial_state'],
            Ar=Ar,
            Br=Br,
            cmd=self.cmd
        )
        adaptive_system = AdaptiveSystem(
            name='adaptive_system',
            initial_state=spec['adaptive_system']['initial_state'],
            A=A,
            B=B,
            gamma1=spec['adaptive_system']['gamma1'],
            gamma2=spec['adaptive_system']['gamma2'],
            Q=spec['adaptive_system']['Q'],
            unc=self.unc
        )

        M_shape = adaptive_system.state_shape[:1] * 2
        N_shape = adaptive_system.state_shape

        self.phi_size = N_shape[0]
        self.y_size = N_shape[1]

        self.observation_space = spaces.Dict({
            "whole_state": infinite_box(
                np.sum((
                    np.shape(spec['main_system']['initial_state']),
                    np.shape(spec['reference_system']['initial_state']),
                    np.shape(np.ravel(spec['adaptive_system']['initial_state'])),
                    np.shape(Br)[1:]
                ))),
            "y": infinite_box(N_shape[1]),
            "phif": infinite_box(N_shape[0])
        })
        self.action_space = spaces.Dict({
            "M": infinite_box(M_shape),
            "N": infinite_box(N_shape)
        })

        super().__init__(
            systems=[
                main_system,
                reference_system,
                adaptive_system,
            ],
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
        x, xr, W, _, _ = states.values()
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

        # info
        info = {
            'time': t,
            'obs': states,
            'reward': reward,
            'done': done,
        }

        # Updates
        self.states = next_states
        self.clock.tick()

        t = self.clock.get()
        obs = self.observation(t, next_states)

        return obs, reward, done, info

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W = states.values()

        e = x - xr
        u = -W.T.dot(self.unc.basis(x))

        xdot = OrderedDict.fromkeys(self.systems)
        xdot.update({
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, x),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e, action),
        })
        return self.unpack_state(xdot)

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False

    def compute_reward(self):
        return None

    def parse_data(self, f):
        data = logger.recursively_load_dict_contents_from_group(f, '/')
        xs = data['state']['main_system']
        Ws = data['state']['adaptive_system']
        u_mrac = np.vstack(
            [-W.T.dot(self.unc.basis(x)) for W, x in zip(Ws, xs)])
        new_data = {
            'control': {
                'MRAC': u_mrac
            }
        }
        new_data['control']['MRAC'] = u_mrac
        return new_data

    def close(self):
        new_data = self.parse_data(self.logger.f)
        logger.save_dict_to_hdf5(self.logger.f, new_data)
        self.logger.close()


class FilterWrapper(Wrapper):
    """
    This wrapper appends two filtering systems to the given env.
    """
    def __init__(self, env):
        super().__init__(env)

        spec = env.spec
        filter_system = FilterSystem(
            name='filter_system',
            initial_state=spec['filter_system']['initial_state'],
            A=spec['reference_system']['Ar'],
            B=spec['main_system']['B'],
            tau=spec['filter_system']['tau']
        )
        filtered_phi = FilteredPhi(
            name='filtered_phi',
            initial_state=spec['filtered_phi']['initial_state'],
            tau=spec['filter_system']['tau'],
            unc=env.unc
        )
        env.append_systems([filter_system, filtered_phi])

        self.observation_space = None
        self.action_space = None

    def reset(self):
        pass

    def step(self, action):
        pass


class MemoryWrapper(Wrapper):
    def __init__(self, env):
        pass

    def reset(self):
        pass

    def step(self, action):
        return self.env.step(action)


class CompositeMRACEnv(BaseEnv):
    def __init__(self, spec):
        A = Ar = spec['reference_system']['Ar']
        B = spec['main_system']['B']
        Br = spec['reference_system']['Br']
        self.mem_max_size = spec['agent']['mem_max_size']
        self.eps = spec['agent']['eps']

        self.obs_logger = logger.Logger(
            file_name=spec['environment']['obs_log_name']
        )

        self.unc = ParamUnc(real_param=spec['main_system']['real_param'])
        self.cmd = SquareCmd(period=20, phase=10)

        main_system = MainSystem(
            name='main_system',
            initial_state=spec['main_system']['initial_state'],
            A=A,
            B=B,
            Br=Br,
            unc=self.unc,
            cmd=self.cmd
        )
        reference_system = RefSystem(
            name='reference_system',
            initial_state=spec['reference_system']['initial_state'],
            Ar=Ar,
            Br=Br,
            cmd=self.cmd
        )
        adaptive_system = AdaptiveSystem(
            name='adaptive_system',
            initial_state=spec['adaptive_system']['initial_state'],
            A=A,
            B=B,
            gamma1=spec['adaptive_system']['gamma1'],
            gamma2=spec['adaptive_system']['gamma2'],
            Q=spec['adaptive_system']['Q'],
            unc=self.unc
        )
        filter_system = FilterSystem(
            name='filter_system',
            initial_state=spec['filter_system']['initial_state'],
            A=A,
            B=B,
            tau=spec['filter_system']['tau']
        )
        filtered_phi = FilteredPhi(
            name='filtered_phi',
            initial_state=spec['filtered_phi']['initial_state'],
            tau=spec['filter_system']['tau'],
            unc=self.unc
        )

        M_shape = adaptive_system.state_shape[:1] * 2
        N_shape = adaptive_system.state_shape

        self.phi_size = N_shape[0]
        self.y_size = N_shape[1]

        self.observation_space = spaces.Dict({
            "whole_state": infinite_box(
                np.sum((
                    np.shape(spec['main_system']['initial_state']),
                    np.shape(spec['reference_system']['initial_state']),
                    np.shape(np.ravel(spec['adaptive_system']['initial_state'])),
                    np.shape(Br)[1:]
                ))),
            "y": infinite_box(N_shape[1]),
            "phif": infinite_box(N_shape[0])
        })
        self.action_space = spaces.Dict({
            "M": infinite_box(M_shape),
            "N": infinite_box(N_shape)
        })

        super().__init__(
            systems=[
                main_system,
                reference_system,
                adaptive_system,
                filter_system,
                filtered_phi,
            ],
            dt=spec['environment']['time_step'],
            max_t=spec['environment']['final_time'],
            ode_step_len=spec['environment']['ode_step_len'],
        )

    def reset(self):
        """
        Resets to the initial states of the systems and initialize the
        memories with those states information.
        """
        states = super().reset()
        t = self.clock.get()
        phi_mem = np.empty((0, self.phi_size))
        y_mem = np.empty((0, self.y_size))
        self.phi_mem, self.y_mem = self.update_mem(states, phi_mem, y_mem)
        return self.observation(t, states, self.phi_mem, self.y_mem)

    def update_mem(self, states, phi_mem, y_mem):
        """
        This function stacks the state informations to the memories,
        and returns them.
        """
        x, xr, W, z, phif = states.values()
        e = x - xr
        y = self.systems['filter_system'].get_y(z, e)
        norm_phif = nla.norm(phif) + self.eps
        phi_mem = np.vstack((phi_mem, phif / norm_phif))
        y_mem = np.vstack((y_mem, y / norm_phif))
        return phi_mem, y_mem

    def observation(self, t, states, phi_mem, y_mem):
        """
        This function converts all the states including the memories to a
        flat array.
        """
        x, xr, W, _, _ = states.values()
        obs = np.hstack((
            x,
            xr,
            W.ravel(),
            self.cmd.get(t),
            np.hstack(phi_mem),
            np.hstack(y_mem)
        ))
        return obs

    def step(self, action):
        states = self.states
        phi_mem = self.phi_mem
        y_mem = self.y_mem
        t = self.clock.get()

        wrapped_action = self.wrap_action(action, phi_mem, y_mem)
        next_states, full_hist = self.get_next_states(t, states, wrapped_action)

        if len(self.phi_mem) == self.mem_max_size:
            argmin = np.argmin(action)
            phi_mem = np.delete(phi_mem, argmin, axis=0)
            y_mem = np.delete(y_mem, argmin, axis=0)

        next_phi_mem, next_y_mem = self.update_mem(next_states, phi_mem, y_mem)

        # Reward
        reward = self.compute_reward(states, wrapped_action['M'])

        # Terminal condition
        done = self.is_terminal()

        # info
        info = {
            'time': t,
            'obs': states,
            'reward': reward,
            'done': done,
            'phi_mem': phi_mem,
            'y_mem': y_mem
        }

        # Updates
        self.states = next_states
        self.phi_mem = next_phi_mem
        self.y_mem = next_y_mem
        self.clock.tick()

        t = self.clock.get()
        obs = self.observation(t, next_states, next_phi_mem, next_y_mem)

        return obs, reward, done, info

    def wrap_action(self, action, phi_mem, y_mem):
        M = mul_dist(action, phi_mem, phi_mem)
        N = mul_dist(action, phi_mem, y_mem)
        action = {'M': M, 'N': N}
        return action

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matrix form.
        """
        x, xr, W, z, phif = states.values()

        e = x - xr
        u = -W.T.dot(self.unc.basis(x))

        xdot = OrderedDict.fromkeys(self.systems)
        xdot.update({
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, x),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e, action),
            'filter_system': self.systems['filter_system'].deriv(z, e, u),
            'filtered_phi': self.systems['filtered_phi'].deriv(phif, x),
        })
        return self.unpack_state(xdot)

    def is_terminal(self):
        if self.clock.get() > self.clock.max_t:
            return True
        else:
            return False

    def compute_reward(self, states, M):
        x, xr, _, _, _ = states.values()
        e = x - xr
        min_eigval = nla.eigvals(M).min()
        e_cost = e.dot(e)
        reward = - 1e2*min_eigval + 1e-2*e_cost
        return reward

    def parse_data(self, f):
        data = logger.recursively_load_dict_contents_from_group(f, '/')
        xs = data['state']['main_system']
        Ws = data['state']['adaptive_system']
        u_mrac = np.vstack(
            [-W.T.dot(self.unc.basis(x)) for W, x in zip(Ws, xs)])
        new_data = {
            'control': {
                'MRAC': u_mrac
            }
        }
        new_data['control']['MRAC'] = u_mrac
        return new_data

    def close(self):
        new_data = self.parse_data(self.logger.f)
        logger.save_dict_to_hdf5(self.logger.f, new_data)
        self.logger.close()


class ParamUnc:
    def __init__(self, real_param):
        self.W = assign_2d(real_param)

    def get(self, state):
        return self.W.T.dot(self.basis(state))

    def basis(self, x):
        return np.hstack((x[:2], np.abs(x[:2])*x[1], x[0]**3))


class SquareCmd:
    def __init__(self, period=2*np.pi, base=0, amplitude=1, duty=0.5, phase=0):
        """
        Parameters
        ----------
        period: float (>0) [sec]
        amplitude: float (>0)
        width: float or int in (0, 100) [%]
        phase: float [sec]
        """
        self.base = base
        self.amplitude = amplitude
        self.duty = duty
        self.phase = phase

        self.modifier = 2 * np.pi / period

    def get(self, t):
        """
        scipy.signal.square:
            The square wave has a period ``2*pi``, has value +1 from 0 to
            ``2*pi*duty`` and -1 from ``2*pi*duty`` to ``2*pi``.
            `duty` must be in the interval [0,1].
        """
        t = t - self.phase
        if t > 0:
            s = scipy.signal.square(self.modifier * t, duty=self.duty)
        else:
            s = 0
        return np.atleast_1d(s + self.base)


class MainSystem(BaseSystem):
    def __init__(self, name, initial_state, A, B, Br, unc, cmd):
        super().__init__(name=name, initial_state=initial_state)
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
    def __init__(self, name, initial_state, Ar, Br, cmd):
        super().__init__(name, initial_state)

        self.Ar = assign_2d(Ar)
        self.Br = assign_2d(Br)
        self.cmd = cmd

    def deriv(self, t, x):
        xdot = self.Ar.dot(x) + self.Br.dot(self.cmd.get(t))
        return xdot


class AdaptiveSystem(BaseSystem):
    def __init__(self, name, initial_state, A, B, gamma1, gamma2, Q, unc):
        super().__init__(name=name, initial_state=initial_state)

        self.A = assign_2d(A)
        self.B = assign_2d(B)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.Q = assign_2d(Q)
        self.P = self.calc_P(self.A, self.B, self.Q)
        self.basis = unc.basis

    def calc_P(self, A, B, Q):
        P = sla.solve_lyapunov(self.A.T, -self.Q)
        return P

    def deriv(self, W, x, e, composite_input):
        M, N = composite_input['M'], composite_input['N']
        Wdot = (
            np.dot(
                self.gamma1, np.outer(self.basis(x), e)
            ).dot(self.P).dot(self.B)
            - np.dot(self.gamma2, M.dot(W) - N)
        )
        return Wdot


class FilterSystem(BaseSystem):
    def __init__(self, name, initial_state, A, B, tau):
        super().__init__(name=name, initial_state=initial_state)
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
    def __init__(self, name, initial_state, tau, unc):
        super().__init__(name=name, initial_state=initial_state)
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


# if __name__ == '__main__':
#     spec = {
#         'experiment': {
#             'name': 'test'
#         },
#         'environment': {
#             'time_step': 0.01,
#             'final_time': 40,
#             'ode_step_len': 3
#         },
#         'main_system': {
#             'initial_state': [0.3, 0, 0],
#             'A': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
#             'B': [[0], [1], [0]],
#             'real_param': [
#                 [-18.59521], [15.162375], [-62.45153], [9.54708], [21.45291]
#             ]
#         },
#         'reference_system': {
#             'initial_state': [0.3, 0, 0],
#             'Ar': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
#             'Br': [[0], [0], [-1]],
#         },
#         'adaptive_system': {
#             'initial_state': [[0], [0], [0], [0], [0]],
#             'gamma': 1000,
#             'Q': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
#         },
#         'filter_system': {
#             'initial_state': [0],
#             'tau': 0.01
#         },
#         'filtered_phi': {
#             'initial_state': [0, 0, 0, 0, 0],
#         },
#         'agent': {
#             'model_log_dir': 'log/model',
#         }
#     }

#     env = CompositeMRACEnv(spec)
#     obs = env.reset()

#     while True:
#         action = {
#             key: np.zeros(space.shape)
#             for key, space in env.action_space.spaces.items()
#         }

#         next_obs, reward, done, info = env.step(action)

#         obs = next_obs

#         if done:
#             break

#     env.close()

#     import matplotlib.pyplot as plt
#     from fym.utils import logger

#     data = logger.load_dict_from_hdf5(env.logger.path)
#     plt.plot(data['time'], data['state']['main_system'][:, :2])
#     plt.show()
