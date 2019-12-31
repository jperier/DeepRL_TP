import random
from collections import OrderedDict

import torch

BUFFER_SIZE = 100000


def create_greedy_exploration(epsilon=0.1):
    def greedy_exploration(q_values, action_space):
        if random.random() < epsilon:
            return action_space.sample()
        return int(torch.argmax(q_values))

    return greedy_exploration


default_exploration = create_greedy_exploration()


class SimpleAgent(object):
    """The world's simplest agent!"""

    def __init__(self, observation_space, action_space, model, gamma=0.95, exploration=default_exploration):
        self.action_space = action_space
        self.buffer = []
        self.index = 0
        self.model = model
        self.gamma = gamma
        self.exploration = exploration

    def act(self, observation, reward, done):
        q_valeurs = self.model.forward(observation)

        # greedy
        return self.exploration(q_valeurs, self.action_space)

    def memorize(self, interaction):
        if len(self.buffer) < BUFFER_SIZE:
            self.buffer.append(interaction)
        else:
            self.buffer[self.index] = interaction
            self.index = (self.index + 1) % BUFFER_SIZE
        assert len(self.buffer) <= BUFFER_SIZE

    def get_batch(self, size=100, repeated=False):
        if len(self.buffer) >= size:
            return random.choices(self.buffer, k=size) if repeated else random.sample(self.buffer, size)
        else:
            return []

    def train(self, batch_size=100):
        batch = self.get_batch(size=batch_size)
        for state, action, next_state, reward, done in batch:
            target = reward  # if done
            if not done:
                # print(batch)
                # print(torch.tensor(next_state).float().shape)
                # print(state, action, reward, next_state, done)
                target = (reward + self.gamma * torch.max(self.model.forward(next_state)))
            target_f = self.model.forward(state)
            target_f[action] = target

            output = self.model.forward(state)
            self.model.backward(output, target_f)

    def save(self, path="/tmp", epoch=1):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'loss': self.model.criterion
        }, path)

    # def load(self):
    #     model = NeuralNetwork(*args, **kwargs)
    #     optimizer = TheOptimizerClass(*args, **kwargs)
    #
    #     checkpoint = torch.load(PATH)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     loss = checkpoint['loss']
    #
    #     model.eval()
    #     # - or -
    #     model.train()
