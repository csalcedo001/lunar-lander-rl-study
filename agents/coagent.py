import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distr
from gym import spaces

from .agent import Agent
from .reinforce import ReinforceAgent


class CoagentNetModel(nn.Module):
    def __init__(self, n_in, n_out, n_h):
        super().__init__()

        layers = [
            nn.Linear(n_in, n_h),
            nn.ReLU(),
        ]
        
        self.base = nn.Sequential(*layers)
        self.mean_net = nn.Sequential(nn.Linear(n_h, n_out), nn.Tanh())
        self.std_net = nn.Sequential(nn.Linear(n_h, n_out), nn.Softplus())
    
    def forward(self, x):
        z = self.base(x)

        mean = self.mean_net(z)
        std = self.std_net(z)

        return mean, std


class CoagentNetAgent(nn.Module):
    def __init__(self, env):
        super().__init__()

        n_in = env.observation_space.shape[0]
        n_out = env.action_space.n

        n_h = 16

        self.model = CoagentNetModel(n_in, n_out, n_h)

        self.onpolicy_reset()
        
    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []
    
    def forward(self, x):
        pdparams = self.model(x)
        return pdparams
    
    def act(self, state):
        pdparams = self.forward(state)
        pdmean, pdstd = pdparams
        
        pd = distr.normal.Normal(pdmean, pdstd)
        action = pd.sample()

        if self.training:
            log_prob = pd.log_prob(action).sum()
            self.log_probs.append(log_prob)

        return action.detach()
    
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


class CoagentNetRelativeEnv():
    def __init__(self, n_in, n_out):
        self.observation_shape = (n_in)
        self.observation_space = spaces.Box(
            high=np.ones(self.observation_shape) * np.inf,
            low=np.ones(self.observation_shape) * -np.inf,
            dtype=np.float32)
        self.action_shape = (n_out)
        self.action_space = spaces.Box(
            high=np.ones(self.action_shape) * np.inf,
            low=np.ones(self.action_shape) * -np.inf,
            dtype=np.float32)
        

class CoagentNetworkAgent(Agent):
    def __init__(
            self,
            env,
            layer_sizes,
            gamma=0.99,
            lr=0.01,
            beta=0.5
        ):

        super().__init__(env)

        n_in = env.observation_space.shape[0]

        if type(env.action_space) == spaces.Discrete:
            n_out = env.action_space.n
        else:
            n_out = env.action_space.shape[0]

        self.layer_sizes = [n_in] + layer_sizes + [n_out]
        self.beta = beta

        self.agents = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = 1 + self.layer_sizes[i + 1]

            agent_relative_env = CoagentNetRelativeEnv(
                n_in=n_in,
                n_out=n_out)

            agent = ReinforceAgent(
                env=agent_relative_env,
                gamma=gamma,
                lr=lr)

            self.agents.append(agent)
        
        self.agents = nn.ModuleList(self.agents)

        self.reset_state()


    def act(self, state):
        self.states[0] = state

        for _ in range(len(self.agents)):
            self.states, self.rewards = self.single_step(
                self.states,
                self.rewards)
        
        return self.states[-1]
    
    def train_start(self, state):
        self.reset_state()
        
        self.states[0] = state

        for i, agent in enumerate(self.agents):
            agent.train_start(self.states[i])

    def train_step(self, state, reward):
        self.states[0] = state
        self.rewards[-1] = reward
        env_reward = reward

        for i, agent in enumerate(self.agents):
            state = self.states[i + 1]
            reward = self.beta * env_reward \
                + (1 - self.beta) * self.rewards[i + 1]

            agent.train_step(state, reward)
    
    def train_end(self, state):
        for i, agent in enumerate(self.agents):
            agent.train_end(self.states[i])
    
    def save(self, save_dict):
        self._save(save_dict)
    
    def load(self, save_dict):
        self._load(save_dict)
    

    def single_step(self, states, rewards):
        next_states = copy.deepcopy(states)
        rewards = copy.deepcopy(rewards)

        for i, agent in enumerate(self.agents):
            action = agent.act(states[i])
            next_state = action[:-1]
            reward = action[-1]

            next_states[i + 1] = next_state
            rewards[i] += reward

        return next_states, rewards
    
    def reset_state(self):
        self.states = []
        self.rewards = []
        for layer_size in self.layer_sizes:
            self.states.append(np.zeros(layer_size))
            self.rewards.append(0)