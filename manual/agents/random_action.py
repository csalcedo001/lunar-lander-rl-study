from .agent import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

        self.action_space = env.action_space

    def act(self, state):
        return self.action_space.sample()