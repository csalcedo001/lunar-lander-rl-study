import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ReinforceAgent(nn.Module):
    def __init__(self, lr=0.001):
        super(ReinforceAgent, self).__init__()

        n_in = 8
        n_out = 4
        
        n_h = 64

        layers = [
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_out),
        ]

        # Remember that this network outputs parameters for
        # a distribution of actions, not actions themselves
        self.model = nn.Sequential(*layers)

        self.train()

    def forward(self, state):
        pdparam = self.model(state)
        return pdparam

    def act(self, state):
        state = torch.from_numpy(state.astype(np.float32))
        # Define the probability distribution by computing
        # the probability distribution parameters with the
        # network
        pdparam = self.forward(state)
        pd = Categorical(probs=None, logits=pdparam)

        # Sample an action according to the probability
        # distribution
        action = pd.sample()

        # Compute the log probability of that action being
        # selected, necessary for backpropagation later
        log_prob = pd.log_prob(action)

        return action.item()


episodes = 10
max_iter = 1000

env = gym.make('LunarLander-v2')
agent = ReinforceAgent()

for episode in range(episodes):
    s = env.reset()
    done = False

    for i in range(max_iter):
        a = agent.act(s)

        s, r, done, _ = env.step(a)
        env.render()

        if done:
            break
