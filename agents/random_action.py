import gym
from argparse import ArgumentParser


class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, state):
        return self.action_space.sample()


parser = ArgumentParser()
parser.add_argument(
    '--episodes',
    type=int, default=10)
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
agent = RandomAgent(env)

for episode in range(episodes):
    s = env.reset()
    done = False

    reward = 0
    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)
        reward += r

        if not no_render:
            env.render()

        if done:
            break

    print('Episode {}. Reward: {}'.format(episode, reward))
