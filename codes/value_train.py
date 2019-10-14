import numpy as np
from collections import OrderedDict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mayavi.mlab as mlab

import torch

from envs import SoaringEnv
from agents import Agent, SafeValue
from utils import Differentiator


def train(save_name):
    np.random.seed(1)
    torch.manual_seed(1)

    env = SoaringEnv(
        initial_state=np.array([0, 0, -5, 13, 0, 0]).astype('float'),
        dt=0.005
    )
    agent = Agent(env, lr=1e-4)

    low = np.hstack((-10, 3, np.deg2rad([-40, -50])))
    high = np.hstack((-1, 15, np.deg2rad([40, 50])))
    dataset = np.random.uniform(low=low, high=high, size=(1000000, 4))
    dataset = torch.tensor(dataset).float()

    agent.dataset = dataset

    agent.train_safe_value(verbose=1)

    # Saving model
    torch.save(agent.safe_value.state_dict(), save_name)


class PlotVar:
    def __init__(self, name, latex, bound, is_angle=False, desc=None):
        self.latex = latex
        self.bound = bound
        self.is_angle = is_angle
        self.desc = desc
        self.name = name

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        self.axis = np.rad2deg(grid) if self.is_angle else grid


def get_vars():
    z = PlotVar(
        name='z', desc='negative altitude',
        latex=r'$z$ (m)', bound=[-10, -0.01],)
    V = PlotVar(
        name='V', desc='air speed',
        latex=r'$V$ (m/s)', bound=[3, 15],)
    gamma = PlotVar(
        name='gamma', desc='negative altitude',
        latex=r'$\gamma$ (deg)', bound=np.deg2rad([-45, 45]), is_angle=True,)
    psi = PlotVar(
        name='psi', desc='negative altitude',
        latex=r'$\psi$ (deg)', bound=np.deg2rad([-60, 60]), is_angle=True,)

    return OrderedDict(z=z, V=V, gamma=gamma, psi=psi)


def gen_grid(pvars, fvars, fvals, num=100):
    grids = np.meshgrid(
        *list(map(lambda x: np.linspace(*x.bound, num), pvars))
    )
    for pvar, grid in zip(pvars, grids):
        pvar.grid = grid

    for fvar, fval in zip(fvars, fvals):
        fvar.grid = np.ones_like(pvars[0].grid) * fval


def draw_safe_value(name, model_file, pvars_name, fvars_name, fvals):
    var_dict = get_vars()
    pvars = [var_dict[pvar_name] for pvar_name in pvars_name]
    fvars = [var_dict[fvar_name] for fvar_name in fvars_name]

    safe_value = SafeValue()
    safe_value.load_state_dict(torch.load(model_file))
    safe_value.eval()

    # Evaluation points (z, V, gamma, psi)
    gen_grid(pvars=pvars, fvars=fvars, fvals=fvals, num=100)

    # Evaluate the safe value for each data point
    s = np.zeros_like(pvars[0].grid)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i, j] = safe_value.from_numpy(
                *map(lambda x: x.grid[i, j], var_dict.values()))

    s = np.ma.array(s, mask=s < 0.)

    # Draw a plot
    fig, ax = plt.subplots(1, 1)
    ax.contour(
        pvars[0].axis, pvars[1].axis, s,
        levels=14, linewidths=0.5, colors='k')
    cntr = ax.contourf(
        pvars[0].axis, pvars[1].axis, s,
        levels=14, cmap='RdBu_r')
    fig.colorbar(cntr, ax=ax)
    ax.set_xlabel(pvars[0].latex)
    ax.set_ylabel(pvars[1].latex)
    fig.tight_layout()
    plt.show()
    fig.savefig(name)


def draw_safe_value_3d(name, model_file, pvars_name, fvars_name, fvals):
    if len(pvars_name) != 3:
        print("Please give 3 variables into pvars!")
        print("Aborted")
        return None

    var_dict = get_vars()
    pvars = [var_dict[pvar_name] for pvar_name in pvars_name]
    fvars = [var_dict[fvar_name] for fvar_name in fvars_name]

    safe_value = SafeValue()
    safe_value.load_state_dict(torch.load(model_file))
    safe_value.eval()

    gen_grid(pvars=pvars, fvars=fvars, fvals=fvals, num=30)

    # Evaluate the safe value for each data point
    # value = np.vectorize(safe_value.from_numpy)
    # s = value(*map(lambda x: x.grid, var_dict.values()))
    s = np.zeros_like(pvars[0].grid)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            for k in range(s.shape[2]):
                s[i, j, k] = safe_value.from_numpy(
                    *map(lambda x: x.grid[i, j, k], var_dict.values()))

    # s = np.ma.array(s, mask=s < 0.)
    print(s.min(), s.max())
    # s[np.logical_or(s < 0., s > 0.9)] = np.nan

    # Draw a plot
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    grids = list(map(lambda x: x.axis, pvars))
    src = mlab.pipeline.scalar_field(s)
    mlab.pipeline.contour_surface(
        src, contours=[0], transparent=False)
    volume = mlab.pipeline.volume(src, vmin=0)
    ranges = np.hstack(list(map(lambda x: [x.min(), x.max()], grids)))
    mlab.axes(
        ranges=ranges,
        xlabel=pvars[0].latex, ylabel=pvars[1].latex, zlabel=pvars[2].latex)
    mlab.colorbar(object=volume, orientation='vertical', title='Safe Value')


if __name__ == '__main__':
    # train('safe-model-v0.0.2.pth')

    # Download: https://drive.google.com/open?id=1-YZ4y3njdqZxpDL6aIFtHHhjuGSJ6Bb8
    model_file = 'safe-model-v0.0.2.pth'

    # draw_safe_value(
    #     name='z-and-V.png', model_file=model_file,
    #     pvars_name=('z', 'V'), fvars_name=('gamma', 'psi'), fvals=(0.25, 0.25))

    # draw_safe_value(
    #     name='gamma-and-psi.png', model_file=model_file,
    #     pvars_name=('gamma', 'psi'), fvars_name=('z', 'V'), fvals=(-2.4, 8))

    draw_safe_value_3d(
        name='gamma-and-psi-z-3d.png', model_file=model_file,
        pvars_name=('gamma', 'psi', 'z'), fvars_name=('V',), fvals=(8,))

    # pointwise_validate(
    #     name='val-gamma-and-psi.png', model_file=model_file,
    #     pvars_name=('gamma', 'psi'), fvars_name=('z', 'V'), fvals=(-2.4, 8))
