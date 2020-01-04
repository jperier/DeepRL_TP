import os

from agents import SimpleAgentStabilized
import matplotlib.pyplot as plt


def train(env, agent, epochs=1000, target_update=10, batch_size=50, render_env=False, batch_training=True):
    rewards = []
    epsilons = []
    for epoch in range(epochs):

        rewards_sum = play_with_env(env, agent, render=render_env)

        rewards.append(rewards_sum)
        eps = agent.get_epsilon()
        epsilons.append(eps)

        if epoch % 100 == 0:
            print(f"episode: {epoch + 1}/{epochs}, score: {rewards_sum}, epsilon: {eps:.2}")
        if len(agent.buffer) > batch_size:
            agent.train(batch_training=batch_training)
        if epoch % target_update == 0 and isinstance(agent, SimpleAgentStabilized):
            agent.update_target_network()

    plot_rewards(rewards, epochs, target_update)


def play_with_env(env, agent, render=False):
    state = env.reset()
    done = False
    count = 0
    rewards_sum = 0
    reward = 0

    while not done:
        if render:
            env.render()
        last_state = state
        action = agent.act(state, reward, done)
        state, reward, done, _ = env.step(action)

        interaction = (last_state, action, state, reward, done)
        agent.memorize(interaction)

        rewards_sum += reward
        if not done:
            count += 1

    return rewards_sum


def plot_rewards(rewards, nb_epoch=None, target_update=None):
    title = "Evolution des récompenses"
    if nb_epoch: title += f', nb_epoch={nb_epoch}'
    if target_update: title += f', target_update={target_update}'
    plt.xlabel("Itérations")
    plt.ylabel("Somme des récompenses")
    plt.title(title)
    plt.plot(rewards)
    plt.show()


def out_dir(outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
