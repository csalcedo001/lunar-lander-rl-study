import os

import pickle
import matplotlib.pyplot as plt
import numpy as np
import xlab.experiment as exp


agent = 'reinforce'
episodes = 3
num_samples = 5


req_args = {
    'agent': agent,
    'episodes': episodes,
    'num_samples': num_samples,
}

e = exp.Experiment(
    executable='train.py',
    req_args=req_args,
    command='python {executable} {agent} --no-render',
)

e.args['no_render'] = True


# e.args['episodes'] = 3
# e.args['num_samples'] = 5

lr_list = [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003025]

losses = []
rewards = []

for lr in lr_list:
    e.args['agent_config']['lr'] = lr

    if not e.is_complete():
        e.run()

    exp_dir = e.get_dir()

    losses_path = os.path.join(exp_dir, 'losses.pkl')
    rewards_path = os.path.join(exp_dir, 'rewards.pkl')

    with open(losses_path, 'rb') as in_file:
        sample_losses = pickle.load(in_file)

    with open(rewards_path, 'rb') as in_file:
        sample_rewards = pickle.load(in_file)
    
    losses.append(sample_losses)
    rewards.append(sample_rewards)

# Shape: (sample, repeat, episode)
losses = np.array(losses)
rewards = np.array(rewards)


# data = lr_list
# mean = rewards.mean(axis=2).mean(axis=1)
# var = data.std(axis=1)    # Standard deviation across episodes
# lower = mean - var
# upper = mean + var

# fig = plt.figure()
# plt.title('Hyperparameter search for learning rate')
# plt.xlabel('Learning rate')
# plt.ylabel('Mean reward')
# plt.fill_between(data, lower, upper[i], alpha=0.3)
# plt.plot(data, mean[i], label=legend_labels[method])
# plt.savefig('hs_lr.png')
# plt.close(fig)




x = np.arange(rewards.shape[1])
mean = rewards.mean(axis=2)
var = rewards.std(axis=2)    # Standard deviation across episodes
lower = mean - var
upper = mean + var

fig = plt.figure()
plt.title('Hyperparameter search for learning rate')
plt.xlabel('Learning rate')
plt.ylabel('Mean reward')
for i in range(len(lr_list)):
    plt.fill_between(x, lower[i], upper[i], alpha=0.3)
for i, lr in enumerate(lr_list):
    plt.plot(x, mean[i], label=str(lr))
plt.legend()
plt.savefig('train_reward_lr.png')
plt.close(fig)