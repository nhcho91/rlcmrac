from collections import deque
import numpy as np
import scipy.optimize as sop

import torch
import torch.nn as nn
import torch.autograd as autograd


class SafeValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def const_func(self, x):
        return -0.1*x[:, 0].unsqueeze(1)

    def forward(self, x):
        return self.const_func(x) - self.model(x)

    def get_grad(self, x, requires_grad=False, create_graph=False):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x).float()
        x.requires_grad_()
        value = self(x)
        value_grad, = autograd.grad(
            value, x, torch.ones_like(value), create_graph=create_graph)
        x.requires_grad_(requires_grad)
        return value_grad

    def from_numpy(self, z, V, gamma, psi):
        return self(torch.tensor([[z, V, gamma, psi]]).float()).detach().numpy()


class Agent:
    # Buffer size determines the number of points used to calculate
    # the derivatives
    buffer_size = 3
    sampling_freq = 10  # [Hz]

    def __init__(self, env, lr):
        # self.env = env
        self.system = env.systems['aircraft']
        self.buffer = deque(maxlen=self.buffer_size)
        self.time_step = env.dt
        self.sampling_interval = int(1 / self.time_step / self.sampling_freq)

        # This model accounts for the function as V(x) = l(x) - model(x)
        self.safe_value = SafeValue()
        self.optimizer = torch.optim.Adam(
            self.safe_value.parameters(), lr=lr)

        # Losses
        self.huber_loss = nn.SmoothL1Loss()
        self.cosine_loss = nn.CosineSimilarity()

    def get_safe(self, x):
        # How we define an optimal safety policy when we have the safety
        # function, or the value function.
        x = torch.tensor(x).float()
        value_grad = self.safe_value.get_grad(
            x.unsqueeze(0), create_graph=True)
        detached_grad = value_grad.detach().squeeze()
        return self.get_optimal_control(x, detached_grad)

    def obj_func(self, x, u, d, value_grad,
                 out=np.ndarray, create_graph=False):
        """
        Parameters
        ----------
        out : type
            output type: ``np.ndarray`` or ``torch.tensor``
        """
        # value_grad = self.safe_value.get_grad(x, create_graph=create_graph)
        dyn = torch.tensor(self.dyn(x, u, d)).float()
        res = value_grad.dot(dyn)
        if out == np.ndarray:
            return res.numpy()
        elif out == torch.tensor:
            return res

    def get_optimal_control(self, x, value_grad,
                            x0=[0.5, 0], method='SLSQP', tol=1e-8, eps=1e-4):
        # Define a function that will be minimized.
        # In this case, ``u`` should maximize the objective function
        func = lambda u: -self.obj_func(x, u, 0, value_grad)
        lb = self.system.control_lower_bound[1:]
        ub = self.system.control_upper_bound[1:]
        bounds = list(zip(lb, ub))
        res = sop.minimize(
            func, x0=x0, bounds=bounds, method=method,
            tol=tol, options=dict(eps=eps))

        if res.success:
            return res.x
        else:
            return None

    def get_optimal_disturbance(self, x, value_grad,
                                x0=0, method='SLSQP', tol=1e-8, eps=1e-4):
        # Find the argumnet minimunm of value dot of d
        func = lambda d: self.obj_func(x, [0.5, 0], d, value_grad)
        _, (_, (_, _, d_true), _) = self.system.wind.get(np.hstack((0, 0, x)))
        bounds = ((d_true - 0.1, d_true + 0.1),)
        res = sop.minimize(
            func, x0, bounds=bounds, method=method,
            tol=tol, options=dict(eps=eps))

        if res.success:
            return res.x
        else:
            return None

    def dyn(self, x, u, d):
        # x: z, V, gamma, psi
        # u: CL, phi
        # d: dWydz
        state = np.hstack((0, 0, x))
        control = np.hstack((0, u))
        external = {
            'wind': (np.zeros(3), (0, np.hstack((0, 0, d)), 0))
        }
        return self.system._raw_deriv(state, 0, control, external)[2:]

    def train_safe_value(self, verbose=0):
        self.safe_value.train()

        # Set DataLoader
        trainloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=32, shuffle=False)

        running_loss = 0.5

        for i, x in enumerate(trainloader, 1):
            value_grad = self.safe_value.get_grad(x, create_graph=True)
            detached_grad = value_grad.detach()

            dyn = torch.zeros_like(value_grad)
            for j in range(len(x)):
                umax = self.get_optimal_control(x[j], detached_grad[j])
                dmin = self.get_optimal_disturbance(x[j], detached_grad[j])

                if umax is None or dmin is None:
                    print('Optimization was failed. We continue!')
                    continue

                dyn[j] = torch.tensor(self.dyn(x[j], umax, dmin)).float()

            loss = self.huber_loss(torch.min(
                self.safe_value.const_func(x) - self.safe_value(x),
                self.cosine_loss(value_grad, dyn).unsqueeze(1)
            ), torch.zeros(x.shape[0], 1))
            # loss = torch.norm(
            #     torch.sum(value_grad * dyn, axis=1)
            #     / (torch.norm(value_grad, dim=1) * torch.norm(dyn, dim=1)
            #        + 1e-8)
            #     # - 0.01*self.safe_value(x)
            # )

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Verbose
            running_loss = 0.9*running_loss + 0.1*loss.item()

            if verbose == 0:
                continue

            if i % 50 == 0:
                print(f'Step: {i:4d} - '
                      f'running loss: {running_loss:10.4f} - '
                      'sample umax: ' + ' '.join(f'{x:.3f}' for x in umax)
                      + f', dmin: {dmin.item():.3f}')

    def get_data(self):
        x, u = np.split(np.array(self.buffer), [6, ], axis=1)
        med_x, med_u = np.median(x, axis=0), np.median(u, axis=0)
        state_deriv = (x[-1] - x[0]) / (self.time_step * (self.buffer_size - 1))

        # Find f(x, u)
        f = self.system._raw_deriv(
            med_x, 0, med_u, {'wind': (np.zeros(3), np.zeros((3, 3)))})

        # Calculate x_dot - f(x, u) = h(x)*d(x)
        # where d(x) = (Wy_hat(x), dWydz_hat(x))
        hd = state_deriv - f

        Wy_hat = hd[1]

        *_, z, V, gamma, psi = med_x
        dWydz_hat = hd[3] / np.cos(gamma) / np.sin(gamma) / np.sin(psi) / V

        return z, (Wy_hat, dWydz_hat)
