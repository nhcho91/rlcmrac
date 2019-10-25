from envs import CmracEnv
from agents import ClAgent
from utils import load_spec
import fym.logging as logging

spec = load_spec('spec.json')
env = CmracEnv(spec)
agent = ClAgent(env)

obs = env.reset()

while True:
    action = agent.act(obs)

    next_obs, reward, done, info = env.step(action)

    if done:
        break

env.save(save_dir='data/mrac')
env.close()
