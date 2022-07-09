import os
import argparse

import pickle
import matplotlib.pyplot as plt
import numpy as np
import xlab.experiment as exp




agent = 'coagent'
episodes = 300
num_samples = 100
run_screen = True

def base_experiment():
    req_args = {
        'agent': agent,
        'episodes': episodes,
        'num_samples': num_samples,
        'env_config': {
            'continuous': True
        }
    }

    e = exp.Experiment(
        executable='train.py',
        req_args=req_args,
        command='python -m train {agent} --no-render',
    )

    return e



# Run learning rate experiments
e = base_experiment()

lr_list = [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003025]
for lr in lr_list:
    print('lr={}'.format(lr))
    e.args['agent_config']['lr'] = lr

    if not e.is_complete():
        if run_screen:
            e.run(custom_command='screen -dmS lr_{} '.format(lr) + e.command)
        else:
            e.run()


# Run gamma experiments
e = base_experiment()
gamma_list = 1 - np.array([0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003025])
for gamma in gamma_list:
    print('gamma={}'.format(gamma))
    e.args['agent_config']['gamma'] = gamma

    if not e.is_complete():
        if run_screen:
            e.run(custom_command='screen -dmS gamma_{} '.format(gamma) + e.command)
        else:
            e.run()

if agent != 'reinforce':
    # Run beta experiments
    e = base_experiment()
    beta_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for beta in beta_list:
        print('beta={}'.format(beta))
        e.args['agent_config']['beta'] = beta

        if not e.is_complete():
            if run_screen:
                e.run(custom_command='screen -dmS beta_{} '.format(beta) + e.command)
            else:
                e.run()



# Run benchmark experiments
e = base_experiment()

agents_list = ['reinforce', 'coagent', 'coagent2']
for agent in agents_list:
    print('agent={}'.format(agent))
    e.args['agent'] = agent

    if not e.is_complete():
        if run_screen:
            e.run(custom_command='screen -dmS agent_{} '.format(agent) + e.command)
        else:
            e.run()
