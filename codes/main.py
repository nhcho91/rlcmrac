"""
Run entire experiments and save thd data
```
python main.py
```
"""
import numpy as np
import sys
import os

import torch

from utils import load_spec

from fym import logging

import envs
import agents


def main(args):
    spec = load_spec("spec.json")
    exp_name = '-'.join([args.env.lower(), args.agent.lower()])
    BASE_LOG_DIR = os.path.join("data", exp_name)

    def parse_data(env, data):
        data = env.data_postprocessing(data)
        path = os.path.join(BASE_LOG_DIR, env.logger.basename)
        logging.save(path, data)

    env = getattr(envs, args.env)(spec, data_callback=parse_data)
    agent = getattr(agents, args.agent)(env, spec)
    logger = logging.Logger(log_dir=BASE_LOG_DIR, file_name='episodic.h5')

    obs = env.reset()

    while True:
        with torch.no_grad():
            action = agent.act(obs)

        next_obs, reward, done, info = env.step(action)

        logger.record(**info)

        obs = next_obs

        if done:
            break

    env.close()
    logger.close()

    data = logging.load(logger.path)
    data = env.data_postprocessing(data)
    logging.save(logger.path, data)

    if args.plot is True:
        import plotting
        plotting.main()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--all', action='store_true')
    group.add_argument(
        '-e', '--env',
        choices=('Mrac', 'Cmrac', 'FeCmrac', 'RlCmrac')
    )
    parser.add_argument(
        "--agent",
        choices=("NullAgent", "SAC"),
        default="NullAgent"
    )
    parser.add_argument(
        '-p', '--plot',
        action='store_true',
        help='If the argument is empty, it will plot the most recent data in'
    )
    args = parser.parse_args()

    main(args)
