import random
from networks import *
import torch

from datetime import datetime

BUFFER_SIZE = 1000000
RAND_SEED = 42
random.seed(RAND_SEED)


class GreedyExploration:
    def __init__(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.999999):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def __call__(self, q_values, action_space):
        return action_space.sample() if random.random() < self.epsilon else int(torch.argmax(q_values))

    def update(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


default_exploration = GreedyExploration()


class SimpleAgent(object):
    """The world's simplest agent!"""

    def __init__(self, observation_space, action_space, create_model_function, gamma=0.99,
                 exploration=default_exploration, device=torch.device('cpu')):
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer = []
        self.index = 0
        self.model = create_model_function().to(device)
        self.gamma = gamma
        self.exploration = exploration
        self._eval = False
        # CUDA
        self.device = device

    def act(self, observation, reward, done):

        q_valeurs = self.model.forward(torch.tensor(observation).float().to(self.device))

        # greedy
        return self.exploration(q_valeurs, self.action_space)

    def eval(self, noTraining=True):  # Remove agent ability to learn
        self._eval = noTraining

    def memorize(self, interaction):
        if self._eval:
            return None
        if len(self.buffer) < BUFFER_SIZE:
            self.buffer.append(interaction)
        else:
            self.buffer[self.index] = interaction
            self.index = (self.index + 1) % BUFFER_SIZE
        assert len(self.buffer) <= BUFFER_SIZE

    def get_batch(self, size=100, repeated=False):
        if len(self.buffer) >= size:
            r = random.choices(self.buffer, k=size) if repeated else random.sample(self.buffer, size)
            return r
        else:
            return []

    def train(self, batch_size=32, batch_training=True):
        if not self._eval:
            batch = self.get_batch(size=batch_size)
            if len(batch) == 0:
                return None

            if batch_training:
                states, actions, next_states, rewards, dones = \
                    map(lambda x: x.to(self.device), map(torch.tensor, zip(*batch)))   # converting in tensors
                # efficace si on est pas souvent done
                targets = (rewards + self.gamma * torch.max(self.target_q_values(next_states)))
                targets = torch.where(dones, rewards, targets)

                actions = F.one_hot(actions, self.action_space.n)   # converting actions in one-hot
                targets = targets.unsqueeze(1) * actions            # replacing ones in actions by the target value
                targets_f = self.model.forward(states.to(self.device))              # computing current model predictions
                targets_f = torch.where(actions == 1, targets, targets_f)   # replacing
                self.fit_model(states, targets_f)

            else:
                for state, action, next_state, reward, done in batch:
                    target = reward  # if done
                    if not done:
                        target = (reward + self.gamma * torch.max(self.target_q_values(next_state)))
                    target_f = self.model.forward(state)
                    target_f[action] = target
                    self.fit_model(state, target_f)

            self.exploration.update()

    def get_epsilon(self):
        return self.exploration.epsilon

    def target_q_values(self, next_state):
        return self.model.forward_no_grad(next_state)

    def fit_model(self, state, target_f):
        output = self.model.forward(state)
        self.model.backward(output, target_f)

    def save(self, path="models/", epoch=1):
        if type(self.model) == NeuralNetwork:
            t = 'nn'
        else:
            t = 'conv'

        path += 'model_save_' + str(datetime.now()).replace(':', '-').replace(' ', '_')

        torch.save({
            'nn_type': t,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            # 'optimizer_state_dict': self.model.optimizer.state_dict(),
            # 'loss': self.model.criterion
        },
            path)

        print('Model saved in', path)
        # return path

    def load(self, path):
        checkpoint = torch.load(path)

        input_dim = 1
        for d in self.observation_space.shape:
            input_dim *= d

        if checkpoint['nn_type'] == 'conv':
            self.model = ConvolutionalNetwork(input_dim, self.action_space.n)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('model loaded, epoch:', checkpoint['epoch'])


class SimpleAgentStabilized(SimpleAgent):

    def __init__(self, observation_space, action_space, create_model_function, gamma=0.99,
                 exploration=default_exploration, device=torch.device('cpu')):
        super().__init__(observation_space, action_space, create_model_function, gamma, exploration, device)
        self.target_net = create_model_function().to(self.device)
        self.update_target_network()
        self.target_net.eval()

    def update_target_network(self):
        self.target_net.clone_from(self.model)

    def target_q_values(self, next_state):
        return self.target_net.forward_no_grad(next_state)

    def load(self, path):
        super().load(path)
        self.update_target_network()
