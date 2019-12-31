import argparse
import os
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

    def __init__(self, observation_space, action_space, model, gamma=0.95, epsilon=0.1):
        self.action_space = action_space
        self.buffer = []
        self.index = 0
        self.model = model
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, observation, reward, done):
        q_valeurs = model.forward(observation)

        # greedy
        if random.random() < self.epsilon:
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

    def train(self):
        batch = self.get_batch()
        for state, action, reward, next_state, done in batch:
            target = reward  # if done
            if not done:
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
        print("Model created: ", self.layers)

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
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    model = NeuralNetwork(input_dim, env.action_space.n)
    agent = SimpleAgent(env.observation_space, env.action_space, model)
    batch_size = 32

    episode_count = 100
    reward = 0

    rewards = []

    for i in range(episode_count):
        # if i % 50 == 0:
        #     print(f"Starting episode {i+1}/{episode_count}")
        ob = env.reset()
        done = False
        count = 0
        summ = 0
        while not done:
            # env.render()
            last_state = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)

            interaction = (last_state, action, ob, reward, done)
            agent.memorize(interaction)

            summ += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(i+1, episode_count, count, agent.epsilon))

            count += 1
            agent.train()
            # if len(agent.buffer) > batch_size:
            #     agent.train()
            # else:
            #     print(f"Agent can't learn, buffer size={len(agent.buffer)}")
            # if i % 50 == 0:
            #     agent.save(path=outdir + "weights_"
            #                + '{:04d}'.format(i) + ".hdf5")
        rewards.append(summ)
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    print(rewards)
    plt.plot(rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
