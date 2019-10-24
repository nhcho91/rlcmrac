"""
Run entire experiments and save thd data
```
python main.py
```
"""
import numpy as np
import sys
import ujson

import torch

from envs import CompositeMRACEnv, MRACEnv
from agents import Void, Agent


class Session:
    def __init__(self, spec):
        self.spec = spec

        # Define agent and env
        self.env = CompositeMRACEnv(spec)
        self.agent = Agent(spec)

    def run(self):
        """Run the main RL loop until clock.max_frame"""
        print(f"Running RL loop")

        state = self.env.reset()

        while True:
            with torch.no_grad():
                action = self.agent.act(state)

            next_state, reward, done, info = self.env.step(action)

            # The agent updates their internal state using
            # all the information we've observed.
            self.agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                break

        self.close()

    def close(self):
        self.env.close()

        import matplotlib.pyplot as plt
        from fym.utils import logger

        data = logger.load_dict_from_hdf5(self.env.logger.path)
        plt.plot(data['time'], data['state']['main_system'][:, :2])
        plt.show()


class Trial:
    """
    Trial construct a set of ``Session``s for the same spec.
    After the running each ``Session``, the results is gathered and analyzed
    to produce trial data.
    """
    def __init__(self, spec):
        self.spec = spec

    def run_sessions(self):
        # TODO:
        #   - embed the parallelize_sessions method (multiprocessing)
        pass

    def run(self):
        Session(self.spec).run()
        # session_metrics_list = self.run_sessions()
        # methrics = analysis.analyze_trial(self.spec, session_metrics_list)
        print('Trial done')
        return None


def run_mrac():
    with open('spec.json', 'r') as f:
        spec = ujson.load(f)

    Trial(spec).run()

    env = MRACEnv(spec)
    agent = Agent(env, spec)

    state = env.reset()

    while True:
        with torch.no_grad():
            action = agent.act(state)

        next_state, reward, done, info = env.step(action)

        # The agent updates their internal state using
        # all the information we've observed.
        agent.update(state, action, reward, next_state, done)

        state = next_state

        if done:
            break

    env.close()


def main(args):
    if args.env == 'mrac':
        env = MRACEnv(spec)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-a', '--all', action='store_true')
    group.add_argument('-e', '--env', choices=('mrac', 'clmrac', 'rlmrac'))
    parser.add_argument(
        '-p', '--plot',
        help='If the argument is empty, it will plot the most recent data in'
    )
    args = parser.parse_args()

    main(args)
