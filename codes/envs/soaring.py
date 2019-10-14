import numpy as np
import gym
from gym import spaces

from fym.core import BaseEnv, BaseSystem


class Wind:
    def __init__(self, Wref=10, href=10, h0=0.03):
        self.Wref = Wref
        self.href = href
        self.h0 = h0

    def get(self, state):
        _, _, z, V, gamma, _ = state
        h = -z

        h = max(h, self.h0)

        Wy = self.Wref*np.log(h/self.h0)/np.log(self.href/self.h0)
        dWydz = -self.Wref/h/np.log(self.href/self.h0)

        vel = [0, Wy, 0]
        grad = [[0, 0, 0], [0, 0, dWydz], [0, 0, 0]]

        return vel, grad


class SoaringEnv(BaseEnv):
    def __init__(self, initial_state, dt=0.01, Wref=10, href=10, h0=0.03):
        wind = Wind(Wref, href, h0)
        aircraft = Aircraft3Dof(initial_state=initial_state, wind=wind)

        obs_sp = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32,
        )
        act_sp = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]),
            high=np.array([-np.inf, -np.inf]),
            dtype=np.float32,
        )

        super().__init__(systems=[aircraft], dt=dt, obs_sp=obs_sp, act_sp=act_sp)

    def reset(self, noise=0):
        super().reset()
        self.states['aircraft'] += np.random.uniform(-noise, noise)
        return self.get_ob()

    def step(self, action):
        controls = dict(aircraft=action)
        # lb = np.array(self.systems['aircraft'].control_lower_bound)
        # ub = np.array(self.systems['aircraft'].control_upper_bound)
        # aircraft_control = (lb + ub)/2 + (ub - lb)/2*np.asarray(action)
        # controls = dict(aircraft=aircraft_control)

        # states = self.states.copy()
        next_obs, reward, done, _ = super().step(controls)
        # info = {'states': states, 'next_states': self.states}
        return next_obs, reward, done, None

    def get_ob(self):
        states = self.states['aircraft']
        return states

    def terminal(self):
        state = self.states['aircraft']
        system = self.systems['aircraft']
        lb, ub = system.state_lower_bound, system.state_upper_bound
        if not np.all([state > lb, state < ub]):
            return True
        else:
            return False

    def get_reward(self, controls):
        state = self.states['aircraft'][2:]
        goal_state = [-5, 10, 0, 0]
        error = self.weight_norm(state - goal_state, [0.02, 0.01, 1, 1])
        return -error

    def weight_norm(self, v, W):
        if np.asarray(W).ndim == 1:
            W = np.diag(W)
        elif np.asarray(W).ndim > 2:
            raise ValueError("W must have the dimension less than or equal to 2")
        return np.sqrt(np.dot(np.dot(v, W), v))


class Aircraft3Dof(BaseSystem):
    g = 9.80665
    rho = 1.2215
    m = 8.5
    S = 0.65
    b = 3.44
    CD0 = 0.033
    CD1 = 0.017
    Tmax = 7
    name = 'aircraft'
    control_size = 3  # T, CL, phi
    state_lower_bound = [-np.inf, -np.inf, -np.inf, 3, -np.inf, -np.inf]
    state_upper_bound = [np.inf, np.inf, -0.01, np.inf, np.inf, np.inf]
    control_lower_bound = [0, -0.5, np.deg2rad(-70)]
    control_upper_bound = [1, 1.5, np.deg2rad(70)]

    def __init__(self, initial_state, wind):
        super().__init__(self.name, initial_state, self.control_size)
        self.wind = wind
        self.term1 = self.rho*self.S/2/self.m

    def external(self, states, controls):
        state = states['aircraft']
        return dict(wind=self.wind.get(state))

    def deriv(self, state, t, control, external):
        return self._raw_deriv(state, t, control, external)

    def _raw_deriv(self, state, t, control, external):
        x, y, z, V, gamma, psi = state
        T, CL, phi = control
        (_, Wy, _), (_, (_, _, dWydz), _) = external['wind']

        CD = self.CD0 + self.CD1*CL**2

        dxdt = V*np.cos(gamma)*np.cos(psi)
        dydt = V*np.cos(gamma)*np.sin(psi) + Wy
        dzdt = - V*np.sin(gamma)

        dWydt = dWydz * dzdt

        dVdt = (self.Tmax*T/self.m - self.term1*V**2*CD - self.g*np.sin(gamma)
                - dWydt*np.cos(gamma)*np.sin(psi))
        dgammadt = (self.term1*V*CL*np.cos(phi) - self.g*np.cos(gamma)/V
                    + dWydt*np.sin(gamma)*np.sin(psi)/V)
        dpsidt = (self.term1*V/np.cos(gamma)*CL*np.sin(phi)
                  - dWydt*np.cos(psi)/V/np.cos(gamma))

        return np.array([dxdt, dydt, dzdt, dVdt, dgammadt, dpsidt])
