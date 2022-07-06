# Import the RL algorithm (Trainer) we would like to use.
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "LunarLander-v2",
    "env_config": {
        "continuous": True,
    },
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
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

ray.init()

tune.run(
    "PPO",
    max_failures=10,
    stop={"episode_reward_mean": 200},
    config=config,
    checkpoint_at_end=True,
    checkpoint_freq=1,
    queue_trials=True,
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