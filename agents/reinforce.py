import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, env, gamma=0.99, lr=0.01):
        super(ReinforceAgent, self).__init__(env)

        n_in = env.observation_space.shape[0]
        n_out = env.action_space.n
        
        n_h = 64

        self.gamma = gamma

        layers = [
            nn.Linear(n_in, n_h),
            # nn.ReLU(),
            # nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_out),
        ]

        # Remember that this network outputs parameters for
        # a distribution of actions, not actions themselves
        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

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
        if self.training:
            log_prob = pd.log_prob(action)
            self.log_probs.append(log_prob)

        return action.item()
    
    def train_start(self, state):
        self.onpolicy_reset()

    def train_step(self, state, reward):
        self.rewards.append(reward)
    
    def train_end(self, state):
        return self.optimize()
    
    def save(self, save_dict):
        self._save(save_dict)
    
    def load(self, save_dict):
        self._load(save_dict)
    

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def optimize(self):
        loss = 0.
        rets = 0.
        for t in reversed(range(len(self.rewards))):
            rets = self.rewards[t] + self.gamma * rets
            loss += -rets * self.log_probs[t]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()