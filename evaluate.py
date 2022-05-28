import gym
import numpy as np
from argparse import ArgumentParser

from agents.reinforce import ReinforceAgent

parser = ArgumentParser()
parser.add_argument(
    '--episodes',
    type=int, default=100)
parser.add_argument(
    '--max-iter',
    type=int, default=1000)
parser.add_argument(
    '--no-render',
    default=False, action="store_const", const=True)

args = parser.parse_args()

episodes = args.episodes
max_iter = args.max_iter
no_render = args.no_render


env = gym.make('LunarLander-v2')
agent = ReinforceAgent()
agent.eval()

for episode in range(episodes):
    s = env.reset()
    done = False

    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)

        agent.rewards.append(r)

        if not no_render:
            env.render()

        if done:
            break

    self.onpolicy_reset()
    reward = np.sum(agent.rewards)
    print('Episode {}. Reward: {}'.format(episode, reward))
