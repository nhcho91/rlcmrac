import numpy as np
import sys

from envs.composite_mrac import CompositeMRACEnv


class Agent:
    def __init__(self):
        pass

    def update(self):
        self.apgorithm.train()
        self.algorithm.update()


class Session:
    def __init__(self, spec):
        self.spec = spec

        # Define agent and env
        self.env = CompositeMRACEnv()
        self.agent = Agent()

    def try_ckpt(self, agent, env):
        """Check then run checkpoint log/eval"""
        pass

    def run(self):
        """Run the main RL loop until clock.max_frame"""
        print(f"Running RL loop")
        clock = self.env.clock
        state = self.env.reset()

        while True:
            self.try_ckpt(self.agent, self.env)

            if clock.get() >= clock.max_frame:
                break

            with torch.no_grad():
                action = self.agent.act(state)

            next_state, reward, done, info = self.env.step(action)

            # The agent updates their internal state using
            # all the information we've observed.
            self.agent.update(state, action, reward, next_state, done)

            state = next_state

            if done:
                pass


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
    args = sys.argv[1:]
    Trial(spec).run()


if __name__ == "__main__":
    main()
