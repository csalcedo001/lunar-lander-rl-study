import gym
import numpy as np
from argparse import ArgumentParser

from agents.reinforce import ReinforceAgent


parser = ArgumentParser()
parser.add_argument(
    '--episodes',
    type=int, default=1000)
parser.add_argument(
    '--max-iter',
    type=int, default=1000)
parser.add_argument(
    '--no-render',
    default=False, action='store_const', const=True)

args = parser.parse_args()

episodes = args.episodes
max_iter = args.max_iter
no_render = args.no_render


env = gym.make('CartPole-v1')
agent = ReinforceAgent(env)

agent.train()

for episode in range(episodes):
    s = env.reset()
    done = False

    agent.train_start(s)

    total_reward = 0.
    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)
        total_reward += r

        agent.train_step(s, r)

        if not no_render:
            env.render()

        if done:
            break
    
    loss = agent.train_end(s)
    print('Episode {}. Loss: {}. Reward: {}'.format(episode, loss, total_reward))

agent.save('.')