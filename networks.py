import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.rmsprop import RMSprop


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
            self.layers.add_module(f"linear{nb_hidden_layers - 1}", nn.Linear(self.hidden_layer_size, output_dim))
        print("Model created: ", self.layers)
        print(f" with input size = {input_dim}/output size = {output_dim}")

    def forward(self, x):
        x = torch.tensor(x).float()
        return self.layers.forward(x)

    def forward_no_grad(self, x):
        with torch.no_grad():
            return self.forward(x)

    def backward(self, output, target):
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

    def clone_from(self, network):
        self.load_state_dict(network.state_dict())


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, nb_hidden_layers=1, hidden_layer_size=None,
                 loss_function=nn.MSELoss):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, output_dim)

        self.criterion = loss_function()
        self.optimizer = RMSprop(self.parameters(),
                                 lr=2.5e-4,
                                 alpha=0.95,
                                 momentum=0.95,
                                 eps=0.01)

    def forward(self, x):
        x = torch.tensor(x).float()
        #print('in', x.shape)
        to_squeeze = False
        if len(x.shape) == 3:
            to_squeeze = True
            x = x.unsqueeze(0)  # adding the batch dim
        #print('post unsqueeze', x.shape)
        x = self.conv1(x)
        #print('post conv1', x.shape)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        #print('post conv2', x.shape)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        #print('post conv3', x.shape)
        x = F.leaky_relu(x)
        x = torch.reshape(x, (x.shape[0], -1))  # flattening but keeping batch dim
        #print('post flatten', x.shape)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # we remove the batch dim if we were given only one instance
        if to_squeeze:
            return x.squeeze(0)
        else:
            return x

    def forward_no_grad(self, x):
        with torch.no_grad():
            return self.forward(x)

    def backward(self, output, target):
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

    def clone_from(self, network):
        self.load_state_dict(network.state_dict())
