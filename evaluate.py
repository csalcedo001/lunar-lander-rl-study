import os

import gym

from agents.reinforce import ReinforceAgent
from agents.random_action import RandomAgent
from parser import get_parser



### Get arguments

parser = get_parser()
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
    class_list = list(agent_name_type_map.values())
    class_names = [cls.__name__ for cls in class_list]

    raise Exception('Invalid agent: choose from {}.'.format(class_names))

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
    raise Exception('Error: checkpoint argument should be a directory.')
    


### Setup for evaluation

env = gym.make(env_name)
agent = agent_class(env)

if not agent_name == 'random' and checkpoint_dir != None:
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
