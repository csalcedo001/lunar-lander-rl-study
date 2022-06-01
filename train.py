import os

import gym
import json
import xlab.experiment as exp

from agents.reinforce import ReinforceAgent
from parser import get_parser



parser = get_parser()

with exp.setup(parser, hash_ignore=['no_render']) as setup:
    ### Get arguments

    args = setup.args
    dir = setup.dir

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

    if load_q and not os.path.isdir(checkpoint_dir):
        raise Exception('Error: checkpoint argument should be a directory.')



    ### Setup for training

    env = gym.make(env_name)
    agent = agent_class(env)

    if checkpoint_dir != None:
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
        print('Episode {}. Loss: {}. Reward: {}'.format(episode, loss, total_reward))

    agent.save(dir)