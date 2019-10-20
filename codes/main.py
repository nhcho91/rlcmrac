import numpy as np
import sys
from tqdm import tqdm

import torch

from envs.composite_mrac import CompositeMRACEnv
from agents import Agent


class Session:
    def __init__(self, spec):
        self.spec = spec

        # Define agent and env
        self.env = CompositeMRACEnv(spec)
        self.agent = Agent(spec)

    def try_ckpt(self, agent, env):
        """Check then run checkpoint log/eval"""
        pass

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


class Trainer():
    pass


def main():
    spec = {
        'experiment': {
            'name': 'test'
        },
        'environment': {
            'time_step': 0.01,
            'final_time': 60,
            'ode_step_len': 3
        },
        'main_system': {
            'initial_state': [0.3, 0, 0],
            'A': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
            'B': [[0], [1], [0]],
            'real_param': [
                [-18.59521], [15.162375], [-62.45153], [9.54708], [21.45291]
            ]
        },
        'reference_system': {
            'initial_state': [0.3, 0, 0],
            'Ar': [[0, 1, 0], [-15.8, -5.6, -17.3], [1, 0, 0]],
            'Br': [[0], [0], [-1]],
        },
        'adaptive_system': {
            'initial_state': [[0], [0], [0], [0], [0]],
            'gamma1': 1000,
            'gamma2': 0,
            'Q': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        },
        'filter_system': {
            'initial_state': [0],
            'tau': 0.001
        },
        'filtered_phi': {
            'initial_state': [0, 0, 0, 0, 0],
        },
        'agent': {
            'mem_max_size': 100,
            'eps': 1e-10,
            'batch_size': 128,
            'hidden_size': 32,
            'buffer_size': 1e4,
            'value_lr': 1e-4,
            'soft_q_lr': 1e-4,
            'policy_lr': 1e-4,
            'model_log_dir': 'log/model',
        }
    }

    Trial(spec).run()


if __name__ == "__main__":
    main()
