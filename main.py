import gym
import numpy

episodes = 10
max_iter = 1000

env = gym.make("LunarLander-v2")

for episode in range(episodes):
    s = env.reset()
    done = False

    for i in range(max_iter):
        a = env.action_space.sample()

        s, r, done, _ = env.step(a)
        env.render()

        if done:
            break
