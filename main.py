"""
Run entire experiments and save thd data
```
python main.py
```
"""
import numpy as np
import sys
import os
import click
import shutil
from tqdm import tqdm

import torch

from fym import logging

import config
import envs


@click.group()
def main():
    pass


@main.command()
@click.option("-p", "--draw-plot", is_flag=True)
def sim1(draw_plot):
    gammalist = [5e2, 1e4]
    kwargs = dict(dt=0.01, max_t=20)
    basedir = os.path.join(config.DATADIR, "sim1")
    if os.path.isdir(basedir):
        shutil.rmtree(basedir)
    os.makedirs(basedir)

    for gamma in tqdm(gammalist):
        env = envs.Sim1(gamma=gamma, **kwargs)
        env.reset()
        path = os.path.join(basedir, f"MRAC-{int(gamma)}.h5")
        logger = logging.Logger(path=path)
        logger.set_info(gamma=gamma)

        while True:
            done, info = env.step()

            logger.record(**info)

            if done:
                break

        env.close()
        logger.close()

    if draw_plot:
        import plot
        plot.draw_keynotefig1()


@main.command()
@click.option("-e", "--selected", type=click.Choice(config.ENVLIST),
              multiple=True)
def sim2(selected):
    if not selected:
        selected = config.ENVLIST

    for selected_envclass in selected:
        for envclass in envs.__dir__():
            if selected_envclass.lower() == envclass.lower():
                break

        env = getattr(envs, envclass)()
        print(env)


def run(env, agent, savepath):
    env.logger = logging.Logger(savepath)
    obs = env.reset()
    while True:
        env.render()

        with torch.no_grad():
            action = agent.act(obs)

        next_obs, reward, done, info = env.step(action)

        agent.update(obs, action, reward, next_obs, done)

        obs = next_obs

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
