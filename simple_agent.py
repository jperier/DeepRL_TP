import argparse
import os

import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt

from agents import SimpleAgent, SimpleAgentStabilized
from networks import NeuralNetwork


def train(env, agent, epochs=1000, target_update=10, batch_size=50):
    rewards = []
    epsilons = []
    for epoch in range(epochs):

        rewards_sum = play_with_env(env, agent)

        rewards.append(rewards_sum)
        eps = agent.get_epsilon()
        epsilons.append(eps)

        if epoch % 100 == 0:
            print(f"episode: {epoch + 1}/{epochs}, score: {rewards_sum}, epsilon: {eps:.2}")
        if len(agent.buffer) > batch_size:
            agent.train()
        if epoch % target_update == 0 and isinstance(agent, SimpleAgentStabilized):
            agent.update_target_network()

    plot_rewards(rewards)


def play_with_env(env, agent):
    state = env.reset()
    done = False
    count = 0
    rewards_sum = 0
    reward = 0

    while not done:
        last_state = state
        action = agent.act(state, reward, done)
        state, reward, done, _ = env.step(action)

        interaction = (last_state, action, state, reward, done)
        agent.memorize(interaction)

        rewards_sum += reward
        if not done:
            count += 1

    return rewards_sum


def plot_rewards(rewards):
    # plt.set_xlabel("Itérations")
    # plt.set_ylabel("Somme des récompenses")
    # plt.set_title("Evolution des récompenses")
    plt.plot(rewards)
    plt.show()


def do(args):
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
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    # Calculating input space size
    input_dim = 1
    for d in env.observation_space.shape:
        input_dim *= d

    def create_wrapper():
        def wrapper(x):
            return x
        return wrapper

    def create_model():
        return NeuralNetwork(input_dim, env.action_space.n)
    agent = SimpleAgentStabilized(env.observation_space, env.action_space, create_model)

    train(env, agent, epochs=10, target_update=50)

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()
    do(args)
