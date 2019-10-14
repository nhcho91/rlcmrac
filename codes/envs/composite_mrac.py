import numpy as np
import scipy.linalg as sla
import scipy.signal
import pydash as ps
from collections import OrderedDict
from gym import spaces

from utils import assign_2d
from fym.core import BaseEnv, BaseSystem, infinite_box


class CompositeMRACEnv(BaseEnv):
    def __init__(self, spec):
        A = Ar = spec['reference_system']['Ar']
        B = spec['main_system']['B']
        Br = spec['reference_system']['Br']

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
            spec=spec,
            main_system=main_system,
            unc=self.unc
        )

        M_shape = adaptive_system.state_shape[:1] * 2
        N_shape = adaptive_system.state_shape

        self.observation_space = spaces.Dict({
            "tracking_error": infinite_box(main_system.state_shape),
            "basis": infinite_box(adaptive_system.state_shape[:1])
        })
        self.action_space = spaces.Dict({
            "M": infinite_box(M_shape),
            "N": infinite_box(N_shape)
        })

        super().__init__(
            systems=[main_system, reference_system, adaptive_system],
            dt=spec['environment']['time_step'],
        )

    def step(self, action):
        states = self.states
        t = self.clock.get()

        next_states = self.get_next_states(t, states, action)

        # Update
        self.states = next_states

        # Reward
        reward = self.compute_reward()

        # Terminal condition
        done = self.is_terminal()

        # info
        info = {}

        return self.observation(next_states), reward, done, info

    def derivs(self, t, states, action):
        """
        The argument ``action`` here is the composite term of the CMRAC which
        has a matric form.
        """
        x, xr, W = (
            states[_]
            for _ in ('main_system', 'reference_system', 'adaptive_system')
        )

        e = x - xr
        u = W.T.dot(self.unc.basis(x))

        xdot = OrderedDict.fromkeys(self.systems)
        xdot.update({
            'main_system': self.systems['main_system'].deriv(t, x, u),
            'reference_system': self.systems['reference_system'].deriv(t, x),
            'adaptive_system': self.systems['adaptive_system'].deriv(
                W, x, e, action)
        })
        return self.unpack_state(xdot)

    def observation(self, x):
        return x

    def is_terminal(self):
        pass

    def compute_reward(self):
        pass


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
    def __init__(self, name, spec, main_system, unc):
        initial_state = np.zeros_like(spec['main_system']['real_param'])

        super().__init__(name=name, initial_state=initial_state)

        self.A = assign_2d(main_system.A)
        self.B = assign_2d(main_system.B)
        self.gamma = spec['adaptive_system']['gamma']
        self.Q = assign_2d(spec['adaptive_system']['Q'])
        self.P = self.calc_P(self.A, self.B, self.Q)
        self.basis = unc.basis

    def calc_P(self, A, B, Q):
        P = sla.solve_lyapunov(self.A.T, -self.Q)
        return P

    def deriv(self, W, x, e, composite_input):
        M, N = composite_input['M'], composite_input['N']
        Wdot = (
            np.dot(
                self.gamma, np.outer(self.basis(x), e)
            ).dot(self.P).dot(self.B)
            + M.dot(W)
            + N
        )
        return Wdot


if __name__ == '__main__':
    spec = {
        'environment': {
            'time_step': 0.01
        },
        'main_system': {
            'A': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
            'B': [[0], [1], [0]],
            'initial_state': [0.3, 0, 0],
            'real_param': [
                [-18.59521], [15.162375], [-62.45153], [9.54708], [21.45291]
            ]
        },
        'reference_system': {
            'Ar': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
            'Br': [[0], [0], [-1]],
            'initial_state': [0.3, 0, 0],
        },
        'adaptive_system': {
            'gamma': 1,
            'Q': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }
    }

    env = CompositeMRACEnv(spec)
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
