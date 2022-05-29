import os
from argparse import ArgumentParser

import gym

class Agent(nn.Module):
    def __init__(self):
        pass
    
    def act(self, state):
        raise NotImplementedError

    def save(self, dict_path):
        raise NotImplementedError

    def load(self, dict_path):
        raise NotImplementedError
