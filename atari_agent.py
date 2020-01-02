import argparse

import gym
from gym import logger, wrappers

from agents import SimpleAgentStabilized
from networks import NeuralNetwork, ConvolutionalNetwork
from training import train, out_dir


def do(args):
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = out_dir('./tmp/atari-agent-results')
    env = wrappers.AtariPreprocessing(env, screen_size=84, frame_skip=4, grayscale_obs=True)
    env = wrappers.FlattenObservation(env)
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    print(input_dim)
    print(env.observation_space.shape)

    def create_model():
        return ConvolutionalNetwork(input_dim, env.action_space.n)
    agent = SimpleAgentStabilized(env.observation_space, env.action_space, create_model)

    train(env, agent, epochs=10, target_update=50, render_env=True)

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()
    do(args)
