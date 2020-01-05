import gym
from gym import logger, wrappers

import torch

from agents import SimpleAgentStabilized, GreedyExploration
from networks import ConvolutionalNetwork, NeuralNetwork
from training import out_dir, play_with_env


def eval(env, agent, epochs, batch_size=32, render_env=True, batch_training=True):

    for epoch in range(epochs):

        rewards_sum = play_with_env(env, agent, render=render_env)
        print('Game', epoch, 'score:', rewards_sum)


def do(env_id):
    logger.set_level(logger.INFO)
    if env_id != 1:
        env = gym.make('BreakoutNoFrameskip-v4')
        outdir = out_dir('./tmp/atari-agent-results')

        env = wrappers.AtariPreprocessing(env, screen_size=84, frame_skip=4, grayscale_obs=True)
        env = wrappers.FrameStack(env, 4)
        env = wrappers.Monitor(env, directory=outdir, force=True)
        env.seed(0)

        # Calculating input space size
        input_dim = 1
        for d in env.observation_space.shape:
            input_dim *= d

        # CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using CUDA device:', device)
        else:
            device = torch.device('cpu')

        def create_model():
            return ConvolutionalNetwork(input_dim, env.action_space.n)
        agent = SimpleAgentStabilized(env.observation_space, env.action_space, create_model, device=device)

        # load
        agent.load('models/model_save_PE_10h')
        agent.eval()
        eval(env, agent, 5)

    else:
        env = gym.make('CartPole-v1')

        outdir = out_dir('./tmp/simple-agent-results')
        env = wrappers.Monitor(env, directory=outdir, force=True)
        env.seed(0)

        # Calculating input space size
        input_dim = 1
        for d in env.observation_space.shape:
            input_dim *= d

        def create_model():
            return NeuralNetwork(input_dim, env.action_space.n)

        agent = SimpleAgentStabilized(env.observation_space, env.action_space, create_model,
                                      exploration=GreedyExploration(0.001, 0.001, 0.999))

        agent.load('models/cartpole')
        agent.eval()
        eval(env, agent, 3)

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    print('Quel agent voulez vous tester ?')
    print('1 - Agent cartpole')
    print('2 - Agent atari')
    do(int(input()))
