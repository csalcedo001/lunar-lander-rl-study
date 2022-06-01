import gym
from argparse import ArgumentParser

from agents.reinforce import ReinforceAgent
from agents.random_action import RandomAgent


parser = ArgumentParser()
parser.add_argument(
    '--episodes',
    type=int, default=100)
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
agent.load('.')
agent.eval()

for episode in range(episodes):
    s = env.reset()
    done = False

    total_reward = 0.
    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)
        total_reward += r

        if not no_render:
            env.render()

        if done:
            break
    
    print('Episode {}. Reward: {}'.format(episode, total_reward))
