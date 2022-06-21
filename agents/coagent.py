import copy
import numpy as np
import torch.nn as nn
from gym import spaces

from .agent import Agent
from .reinforce import ReinforceAgent



class CoagentNetRelativeEnv():
    def __init__(self, n_in, n_out, action_type):
        self.observation_shape = (n_in)
        self.observation_space = spaces.Box(
            high=np.ones(self.observation_shape) * np.inf,
            low=np.ones(self.observation_shape) * -np.inf,
            dtype=np.float32)
        
        self.action_shape = (n_out)
        if action_type == 'continuous':
            self.action_space = spaces.Box(
                high=np.ones(self.action_shape) * np.inf,
                low=np.ones(self.action_shape) * -np.inf,
                dtype=np.float32)
        elif action_type == 'discrete':
            self.action_space = spaces.Discrete(self.action_shape)
        else:
            raise Exception('Invalid action type.')
        

class CoagentNetworkAgent(Agent):
    def __init__(
            self,
            env,
            layer_sizes,
            gamma=0.99,
            lr=0.01,
            beta=0.5,
            action_type='continuous',
        ):

        super().__init__(env)

        n_in = env.observation_space.shape[0]

        if type(env.action_space) == spaces.Discrete:
            n_out = env.action_space.n
        else:
            n_out = env.action_space.shape[0]

        self.layer_sizes = [n_in] + layer_sizes + [n_out]
        self.beta = beta
        self.action_type = action_type

        self.agents = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = 1 + self.layer_sizes[i + 1]

            agent_relative_env = CoagentNetRelativeEnv(
                n_in=n_in,
                n_out=n_out,
                action_type=action_type)

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
        self.states[0] = state

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
            if self.action_type == 'discrete':
                action = np.squeeze(np.eye(self.layer_sizes[i + 1])[action])
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


class CoagentNetworkAgent2(Agent):
    def __init__(
            self,
            env,
            layer_sizes,
            gamma=0.99,
            lr=0.01,
            beta=0.5,
            action_type='continuous',
        ):

        super().__init__(env)

        n_in = env.observation_space.shape[0]

        if type(env.action_space) == spaces.Discrete:
            n_out = env.action_space.n
        else:
            n_out = env.action_space.shape[0]

        self.layer_sizes = [n_in] + layer_sizes + [n_out]
        self.beta = beta
        self.action_type = action_type

        self.agents = []
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            agent_relative_env = CoagentNetRelativeEnv(
                n_in=n_in,
                n_out=n_out,
                action_type=action_type)

            agent = ReinforceAgent(
                env=agent_relative_env,
                gamma=gamma,
                lr=lr)

            self.agents.append(agent)
        
        self.agents = nn.ModuleList(self.agents)

        self.reset_state()


    def act(self, state):
        self.states[0] = state

        while self.turn != len(self.agents) - 1:
            self.agent_turn()
        self.agent_turn()
        
        return self.states[-1]
    
    def train_start(self, state):
        self.reset_state()

        for i, agent in enumerate(self.agents):
            agent.train_start(self.states[i])

        # for i, agent in enumerate(self.agents):
        #     self.states[i] = state
        #     next_state = agent.act(state)
        #     agent.train_start(state)
        #     state = next_state

    def train_step(self, state, reward):
        self.states[0] = state

        for i in range(len(self.agents) - 1):
            agent = self.agents[i]
            agent.train_step(self.states[i], reward)
            self.agent_turn()
        
        i = len(self.agents) - 1
        agent = self.agents[i]
        agent.train_step(self.states[i], reward)


        # while self.timestep == 0:
        #     self.states[self.turn] = state
        #     agent.train_step(state, reward)
        #     state = agent.act(state)
        #     self.advance_turn()

        # for i, agent in enumerate(self.agents):
        #     self.states[i] = state
        #     agent.train_step(state, reward)
        #     state = agent.act(state)
    
    def train_end(self, state):
        for i, agent in enumerate(self.agents):
            agent.train_end(self.states[i])
    
    def save(self, save_dict):
        self._save(save_dict)
    
    def load(self, save_dict):
        self._load(save_dict)
    
    def reset_state(self):
        self.timestep = 0
        self.turn = 0

        self.states = []
        for layer_size in self.layer_sizes:
            self.states.append(np.zeros(layer_size))
    
    def agent_turn(self):
        i = self.turn
        agent = self.agents[i]

        action = agent.act(self.states[i])
        if self.action_type == 'discrete':
            action = np.squeeze(np.eye(self.layer_sizes[i + 1])[action])
        self.states[i + 1] = action

        self.turn = (self.turn + 1) % len(self.agents)
        if self.turn == 0:
            self.timestep += 1