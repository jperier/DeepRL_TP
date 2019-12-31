import argparse
import random
from collections import OrderedDict
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

BUFFER_SIZE = 100000


def greedy_exploration(q_values, action_space):
    if random.random() < 0.1:
        return action_space.sample()

    return int(torch.argmax(q_values))


def boltzmann_exploration(q_values, action_space):
    i = 42
    # TODO


class SimpleAgent(object):
    """The world's simplest agent!"""

    def __init__(self, observation_space, action_space, model):
        self.action_space = action_space
        self.buffer = []
        self.index = 0
        self.model = model

    def act(self, observation, reward, done):
        q_valeurs = model.forward(observation)

        # greedy
        if random.random() < 0.1:
            return self.action_space.sample()

        return int(torch.argmax(q_valeurs))

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

    # interaction = [last_state, action, new_state, reward, done]
    def learn(self):
        batch = self.get_batch()
        for interaction in batch:
            y = model.forward(interaction[0])
            # model.backward(y, label)

    def bellman(self):
        bouh = 42


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, nb_hidden_layers=1, hidden_layer_size=None, learning_rate=1e-3,
                 loss_function=nn.MSELoss, optimizer=optim.SGD):
        super(NeuralNetwork, self).__init__()
        self.layers = []
        self.hidden_layer_size = hidden_layer_size if hidden_layer_size else int((input_dim + output_dim) / 2)
        self._create_network_(nb_hidden_layers, output_dim)
        self.criterion = loss_function()
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def _create_network_old_(self, nb_hidden_layers, output_dim):
        # Creating layers
        if nb_hidden_layers == 0:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First hidden layer
            self.layers.append(nn.Linear(input_dim, self.hidden_layer_size))
            # Other hidden layers
            for i in range(1, nb_hidden_layers - 1):
                self.layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
            # Output layer
            self.layers.append(nn.Linear(self.hidden_layer_size, output_dim))

    def _create_network_(self, nb_hidden_layers, output_dim):
        layers = []
        if nb_hidden_layers == 0:
            layers.append(('linear', nn.Linear(input_dim, output_dim)))
        else:
            # First hidden layer
            layers.append(('linear1', nn.Linear(input_dim, self.hidden_layer_size)))
            layers.append(("leaky_relu1", nn.LeakyReLU()))
            # Other hidden layers
            for i in range(1, nb_hidden_layers - 1):
                layers.append((f"linear{i}", nn.Linear(self.hidden_layer_size, self.hidden_layer_size)))
                layers.append((f"leaky_relu{i}", nn.LeakyReLU()))
            # Output layer
            layers.append((f"linear{nb_hidden_layers-1}", nn.Linear(self.hidden_layer_size, output_dim)))

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = torch.tensor(x).float()
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))
        # Last Layer without relu
        return self.layers[-1](x)

    def backward(self, output, target):
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './tmp/simple-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    model = NeuralNetwork(input_dim, env.action_space.n)
    agent = SimpleAgent(env.observation_space, env.action_space, model)

    episode_count = 100
    reward = 0
    done = False

    rewards = []

    for i in range(episode_count):
        ob = env.reset()
        count = 0
        summ = 0
        while True:
            last_state = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            interaction = [last_state, action, ob, reward, done]
            agent.memorize(interaction)

            count += 1
            summ += reward
            if done:
                rewards.append(summ)
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    plt.plot(rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()