from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, nb_hidden_layers=1, hidden_layer_size=None, learning_rate=1e-3,
                 loss_function=nn.MSELoss, optimizer=optim.SGD):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential()
        self.hidden_layer_size = hidden_layer_size if hidden_layer_size else int((input_dim + output_dim) / 2)
        self._create_network_(input_dim, nb_hidden_layers, output_dim)
        # self.add_module('layers', self.layers)  # FIXME Useful ??
        self.criterion = loss_function()
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def _create_network_(self, input_dim, nb_hidden_layers, output_dim):
        if nb_hidden_layers < 1:  # Just to be sure
            self.layers.add_module('linear', nn.Linear(input_dim, output_dim))
        else:
            # First hidden layer
            self.layers.add_module('linear1', nn.Linear(input_dim, self.hidden_layer_size))
            self.layers.add_module("leaky_relu1", nn.LeakyReLU())
            # Other hidden layers
            for i in range(1, nb_hidden_layers - 1):
                self.layers.add_module(f"linear{i}", nn.Linear(self.hidden_layer_size, self.hidden_layer_size))
                self.layers.add_module(f"leaky_relu{i}", nn.LeakyReLU())
            # Output layer
            self.layers.add_module(f"linear{nb_hidden_layers-1}", nn.Linear(self.hidden_layer_size, output_dim))
        print("Model created: ", self.layers)
        print(f" with input size = {input_dim}/output size = {output_dim}")

    def forward(self, x):
        x = torch.tensor(x).float()
        return self.layers.forward(x)

    def backward(self, output, target):
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

    def clone_from(self, network):
        self.load_state_dict(network.state_dict())


class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, input_dim, output_dim, nb_hidden_layers=1, hidden_layer_size=None, learning_rate=1e-3,
                 loss_function=nn.MSELoss, optimizer=optim.SGD):
        super().__init__(15, output_dim, nb_hidden_layers, hidden_layer_size, learning_rate, loss_function,
                         optimizer)
        self.conv_net = nn.Sequential()
        self._create_conv_network_(input_dim, 15)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def _create_conv_network_(self, conv_shape, input_dim, conv_nb_hidden_layers=2):
        self.conv_net.add_module('conv1', nn.Conv1d(conv_shape, int(conv_shape/4), 30))
        conv_shape = int(conv_shape/4)
        self.conv_net.add_module('conv2', nn.Conv1d(conv_shape, int(conv_shape/4), 30))
        conv_shape = int(conv_shape / 4)
        self.conv_net.add_module('conv3', nn.Conv1d(conv_shape, input_dim, 30))

    def forward(self, x):
        x = torch.tensor(x).float()
        x = self.conv_net.forward(x)
        return self.layers.forward(x)
