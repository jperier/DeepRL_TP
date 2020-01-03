import argparse

import gym
from gym import wrappers, logger

from agents import SimpleAgent, SimpleAgentStabilized
from networks import NeuralNetwork
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
    outdir = out_dir('./tmp/simple-agent-results')
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    def create_model():
        return NeuralNetwork(input_dim, env.action_space.n)
    agent = SimpleAgentStabilized(env.observation_space, env.action_space, create_model)

    train(env, agent, epochs=20000, target_update=500)

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()
    do(args)
