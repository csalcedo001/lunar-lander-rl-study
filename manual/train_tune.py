from argparse import Namespace
import os

import gym
from ray import tune
import xlab.experiment as exp
from xlab.utils import merge_dicts

from parser import get_parser
from utils import get_config_from_string
from agents.reinforce import ReinforceAgent
from agents.coagent import CoagentNetworkAgent, CoagentNetworkAgent2



def train(config):
    args = Namespace(**config)

    ### Get arguments

    # Main arguments
    env_name = args.env
    agent_name = args.agent
    checkpoint = args.checkpoint

    # Optional arguments
    episodes = args.episodes
    max_iter = args.max_iter
    no_render = args.no_render
    env_config = args.env_config
    agent_config = args.agent_config



    ### Process arguments

    # Validate and get agent class
    agent_name_type_map = {
        'reinforce': ReinforceAgent,
        'coagent': CoagentNetworkAgent,
        'coagent2': CoagentNetworkAgent2,
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
    if checkpoint == None and agent_name !='random':
        warning_msg = "Warning: no checkpoint was provided for agent '{}'."
        print(warning_msg.format(agent_name))

    if checkpoint != None:
        if agent_name == 'random':
            error_msg = 'Error: random agents cannot load from a checkpoint.'
            raise Exception(error_msg)
        
        if type(checkpoint) == str and os.path.isdir(checkpoint):
            checkpoint_dir = checkpoint
        else:
            checkpoint_dict = get_config_from_string(checkpoint)
                
            executable = 'train.py'
            command = 'python -m train {agent}'
            req_args = {
                'agent': agent_name,
                'env': env_name,
            }

            checkpoint_config = merge_dicts(req_args, checkpoint_dict)
            e = exp.Experiment(executable, checkpoint_config, command=command)

            checkpoint_dir = e.get_dir()

        if not os.path.isdir(checkpoint_dir):
            error_msg = "Error: could not load checkpoint from '{}'."
            raise Exception(error_msg.format(checkpoint_dir))



    ### Setup for training

    env = gym.make(env_name, **env_config)
    agent = agent_class(env, **agent_config)

    if checkpoint != None:
        agent.load(checkpoint_dir)

    agent.train()

    for episode in range(episodes):
        s = env.reset()
        done = False

        agent.train_start(s)

        total_reward = 0.
        for i in range(max_iter):
            a = agent.act(s)

            s, r, done, _ = env.step(a)
            total_reward += r

            agent.train_step(s, r)

            if not no_render:
                env.render()

            if done:
                break
        
        loss = agent.train_end(s)
        print('Episode {}. Loss: {}. Reward: {}'.format(
            episode, loss, total_reward))

        return {
            "loss": loss,
            "episode_reward_mean": total_reward,
        }


    # agent.save(dir)


parser = get_parser()
args = parser.parse_args()
config = dict(vars(args))
config['lr'] = tune.loguniform(0.0001, 0.01)

tune.run(
    train,
    name='thesis_001',
    config=config,
    num_samples=10,
    stop={"episode_reward_mean": 100},
)
