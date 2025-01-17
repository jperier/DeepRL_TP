import argparse

import gym
from gym import logger, wrappers

import torch

from agents import SimpleAgentStabilized
from networks import ConvolutionalNetwork
from training import train, out_dir

from time import time

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

    # FIXME Inverse AtariPreprocessing and FrameStack ?
    """ 
    If FrameStack simply do env.step() 4 times and AtariPreprocessing skip 4 frames each times,
     we may be skipping more frames than we want, an I right ?
    """
    env = wrappers.AtariPreprocessing(env, screen_size=84, frame_skip=4, grayscale_obs=True)
    env = wrappers.FrameStack(env, 4)
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    print("Observation space:")
    print(input_dim)
    print(env.observation_space.shape)
    print("Action space:")
    print(env.action_space)

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

    epochs = 1000
    start = time()
    train(env, agent, epochs=epochs, target_update=250, render_env=False)
    end = time()
    print("Training time:", end-start)
    agent.save(epoch=epochs)

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()
    do(args)
