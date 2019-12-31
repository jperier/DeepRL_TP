import argparse
import os
import random

import torch

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

from agents import SimpleAgent
from networks import NeuralNetwork


def greedy_exploration(q_values, action_space):
    if random.random() < 0.1:
        return action_space.sample()

    return int(torch.argmax(q_values))


def boltzmann_exploration(q_values, action_space):
    i = 42
    # TODO


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
    batch_size = 42

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
                print(f"episode: {i+1}/{episode_count}, score: {count}")

            count += 1
            if len(agent.buffer) > batch_size:
                agent.train()
            # if i % 50 == 0:
            #     agent.save(path=outdir + "weights_"
            #                + '{:04d}'.format(i) + ".hdf5")
        rewards.append(summ)
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    plt.plot(rewards)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
