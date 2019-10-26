import numpy as np

from envs import CmracEnv
from agents import SAC
from utils import load_spec

import torch

import fym.logging as logging


def parse_data(env, data):
    xs = data['state']['main_system']
    Ws = data['state']['adaptive_system']

    cmd = np.hstack([env.cmd.get(t) for t in data['time']])
    u_mrac = np.vstack(
        [-W.T.dot(env.unc.basis(x)) for W, x in zip(Ws, xs)])

    data.update({
        "control": {
            "MRAC": u_mrac,
        },
        "cmd": cmd,
    })
    logging.save('data/rlcmrac/history.h5', data)


def main():
    spec = load_spec('spec.json')
    env = CmracEnv(spec, data_callback=parse_data)
    agent = SAC(env, spec)
    logger = logging.Logger(log_dir='data/rlcmrac', file_name='action.h5')

    obs = env.reset()

    while True:
        with torch.no_grad():
            action = agent.act(obs)

        next_obs, reward, done, info = env.step(action)

        logger.record(info)

        obs = next_obs

        if done:
            break

    env.close()
    logger.close()


if __name__ == '__main__':
    main()
