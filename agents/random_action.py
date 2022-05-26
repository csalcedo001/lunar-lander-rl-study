import numpy
import gym

class RandomAgent():
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, state):
        return self.action_space.sample()

episodes = 10
max_iter = 1000

env = gym.make('LunarLander-v2')
agent = RandomAgent(env)

for episode in range(episodes):
    s = env.reset()
    done = False

    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)
        env.render()

        if done:
            break
