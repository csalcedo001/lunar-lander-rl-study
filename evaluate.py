import os
from argparse import ArgumentParser

import gym

from agents.reinforce import ReinforceAgent
from agents.random_action import RandomAgent


parser = ArgumentParser()
parser.add_argument(
    'agent',
    type=str)
parser.add_argument(
    '--env',
    type=str, default='LunarLander-v2')
parser.add_argument(
    '--checkpoint',
    type=str, default=None)
parser.add_argument(
    '--episodes',
    type=int, default=100)
parser.add_argument(
    '--max-iter',
    type=int, default=1000)
parser.add_argument(
    '--no-render',
    default=False, action='store_const', const=True)


### Get arguments

args = parser.parse_args()

# Main arguments
env_name = args.env
agent_name = args.agent
checkpoint_dir = args.checkpoint

# Optional arguments
episodes = args.episodes
max_iter = args.max_iter
no_render = args.no_render

### Process arguments

# Validate and get agent class
agent_name_type_map = {
    'reinforce': ReinforceAgent,
    'random': RandomAgent,
}

if agent_name not in agent_name_type_map:
    raise Exception('Invalid agent: choose from {}.'.format(
        list(agent_name_type_map.values())))

agent_class = agent_name_type_map[agent_name]


# Validate env
valid_envs = [
    'CartPole-v1',
    'LunarLander-v2',
]

if env_name not in valid_envs:
    raise Exception('Invalid environment: choose from {}.'.format(
        valid_envs))


# Validate checkpoint for agent type
load_q = checkpoint_dir != None
if load_q and agent_name == 'random':
    raise Exception('Error: random agents cannot load from a checkpoint.')

if load_q and not os.path.isdir(checkpoint_dir):
    raise Exception('TypeError: checkpoint argument should be a directory.')
    


### Setup for evaluation

env = gym.make(env_name)
agent = agent_class(env)

if not agent_name == 'random'and checkpoint_dir != None:
    agent.load(checkpoint_dir)

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
