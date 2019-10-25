from envs import MracEnv
from agents import NullAgent
from utils import load_spec
import fym.logging as logging

spec = load_spec('spec.json')
env = MracEnv(spec, log_dir='data/mrac')
agent = NullAgent(env)
# logger = logging.Logger(log_dir='log/mrac')

obs = env.reset()

while True:
    action = agent.act(obs)

    next_obs, reward, done, info = env.step(action)

    # logger.record({
    #     'time': env.clock.get(),
    #     'obs': obs,
    # })

    if done:
        break

env.close()

# logger.close()
