import argparse
import os

# Import the RL algorithm (Trainer) we would like to use.
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import json
import pickle
from xlab.utils import merge_dicts


def on_episode_end(info):
    pass

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "num_gpus": 1,
    "num_workers": 8,
    # "lr": tune.loguniform(5e-6, 0.003),
    # "train_batch_size": 256,
    "env": "LunarLander-v2",
    "env_config": {
    #     "continuous": True,
    },
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [16, 16],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_duration": 10,
    "evaluation_config": {
        # "render_env": True,
    },
}

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="")

    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint from which to start training.")
    parser.add_argument("--cluster", default=None, help="Which Ray cluster to connect to.")
    parser.add_argument("--num-cpus", default=50, type=int)
    parser.add_argument("--num-gpus", default=4, type=int)
    parser.add_argument("--config", default=[], action='append')
    return parser

def get_config(args, default_config):
    config = {}
    for config_arg in args.config:
        try:
            config_data = json.loads(config_arg)
        except:
            # If json.loads failed, the input should be a path
            extension = os.path.splitext(config_arg)[-1][1:]

            if extension == 'json':
                loader = json
                open_type = 'r'
            elif extension == 'pkl':
                loader = pickle
                open_type = 'rb'
            else:
                raise Exception("Unsupported extension")

            try:
                with open(config_arg, open_type) as in_file:
                    config_data = loader.load(in_file)
            except:
                raise Exception("Invalid config argument")

        config = merge_dicts(config, config_data)

    config = merge_dicts(default_config, config)

    return config


args = create_parser().parse_args()

config = get_config(args, config)

if args.checkpoint != None:
    checkpoint_dir = os.path.dirname(args.checkpoint)
    params_path = os.path.join(checkpoint_dir, '..', 'params.json')
    with open(params_path, 'r') as in_file:
        params_config = json.load(params_path)

    config = merge_dicts(config, params_config)

tune.run(
    "PPO",
    num_samples=10,
    name="lunar-lander",
    max_failures=10,
    restore=args.checkpoint,
    stop={"timesteps_total": 1000000},
    config=config,
    checkpoint_at_end=True,
    checkpoint_freq=1,
)





# # Create our RLlib Trainer.
# trainer = PPOTrainer(config=config)

# # Run it for n training iterations. A training iteration includes
# # parallel sample collection by the environment workers as well as
# # loss calculation on the collected batch and a model update.
# for i in range(5):
#     print('Episode {}: Reward: '.format(i))
#     trainer.train()

#     if i % 20 == 0:
#         trainer.save()

# # Evaluate the trained Trainer (and render each timestep to the shell's
# # output).
# # trainer.evaluate()
