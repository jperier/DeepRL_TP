import random

import torch

BUFFER_SIZE = 100000
RAND_SEED = 42
random.seed(RAND_SEED)


class GreedyExploration:
    def __init__(self, epsilon=1.0, epsilon_min=0.001, epsilon_decay=0.999):
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

    def __init__(self, observation_space, action_space, create_model_function, gamma=0.95,
                 exploration=default_exploration):
        self.action_space = action_space
        self.buffer = []
        self.index = 0
        self.model = create_model_function()
        self.gamma = gamma
        self.exploration = exploration
        self._eval = False

    def act(self, observation, reward, done):
        q_valeurs = self.model.forward(observation)

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
            return random.choices(self.buffer, k=size) if repeated else random.sample(self.buffer, size)
        else:
            return []

    def train(self, batch_size=32):
        if not self._eval:
            batch = self.get_batch(size=batch_size)
            if len(batch) == 0:
                return None
            for state, action, next_state, reward, done in batch:
                target = reward  # if done
                if not done:
                    target = (reward + self.gamma * torch.max(self.target_q_values(next_state)))
                target_f = self.model.forward(state)
                target_f[action] = target
                self.fit_model(state, target_f)
            self.exploration.update()

        # # début de code pour utiliser des batchs durant l'apprentissage
        # states, actions, next_states, rewards, dones = map(torch.tensor, zip(*batch))
        # # efficace si on est pas souvent done
        # targets = (rewards + self.gamma * torch.max(self.target_q_values(next_states)))
        # targets = torch.where(dones, rewards, targets)
        #
        # targets_f = self.model.forward(states)
        # targets_f[actions] = targets
        # self.fit_model(states, targets_f)

    def get_epsilon(self):
        return self.exploration.epsilon

    def target_q_values(self, next_state):
        return self.model.forward_no_grad(next_state)

    def fit_model(self, state, target_f):
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


class SimpleAgentStabilized(SimpleAgent):

    def __init__(self, observation_space, action_space, create_model_function, gamma=0.95, exploration=default_exploration):
        super().__init__(observation_space, action_space, create_model_function, gamma=0.95, exploration=default_exploration)
        self.target_net = create_model_function()
        self.update_target_network()
        self.target_net.eval()

    def update_target_network(self):
        self.target_net.clone_from(self.model)

    def target_q_values(self, next_state):
        return self.target_net.forward_no_grad(next_state)
