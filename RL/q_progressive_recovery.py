from deep_q_network import DeepQNetwork
from rl_environment import environment
import networkx as nx
from graph_helper import r_graph, r_2d_graph, r_tree, get_root, DP_optimal, plot_graph, simulate_tree_recovery, \
    plot_bar_x, read_gml
import numpy as np
import random
import itertools
from ratio_heuristic import ratio_heuristic
from random_heuristic import random_heuristic
import time


def generate_graph(nodes, utils=[1, 4], demands=[1, 2], load_dir=None, type='random_tree'):
    # Generate random tree
    if type == 'random_tree':
        if load_dir:
            graph = nx.read_gpickle('experiments/{0}_rtree.gpickle'.format(nodes))
        else:
            graph = r_tree(nodes, utils, demands)
        save = 'experiments/{0}_rtree.txt'.format(nodes)
        real_node_num = nodes

    # Generate random graph by adding edges according to some p-value (0.2)
    elif type == 'random_graph':
        if load_dir:
            graph = nx.read_gpickle('experiments/{0}_rgraph.gpickle'.format(nodes))
        else:
            graph = r_graph(num_nodes, 0.2, utils, demands)
        save = 'experiments/{0}_rgraph.txt'.format(nodes)
        real_node_num = nodes

    # Generate random nodes x nodes 2-d grid graph
    elif type == 'grid':
        if load_dir:
            graph = nx.read_gpickle('experiments/{0}x{0}.gpickle'.format(nodes))
        else:
            graph = r_2d_graph(nodes, nodes, utils, demands)
        save = 'experiments/{0}x{0}.gpickle'.format(nodes)
        real_node_num = nodes ** 2

    # Real number of nodes may be different from input (in the case of grid graph)
    return graph, save, real_node_num


def main():
    # Load checkpoint
    load_path = "weights/weights.ckpt"
    save_path = "weights/weights.ckpt"

    # Generate graph for training
    resources = 1
    G, reward_save, num_nodes = generate_graph(nodes=40, type='grid')

    # Read GML (DIGEX Graph)
    # G = read_gml('../gml/DIGEX.gml')
    # num_nodes = len(G)
    # resources = 1

    # Try plotting. If on ssh, don't bother since there are some necessary plt.draw() commands
    # to plot a networkx graph.
    try:
        plot_graph(G, get_root(G), 'rl_graph.png')
    except:
        print('No display')

    # Always use root = 2 for consistency
    root = 2
    env = environment(G, [root], resources)
    n_y = len(env.actions_permutations)
    print('num_edges:', G.number_of_edges())
    print("Ratio Heuristic", ratio_heuristic(G, [root], resources), '\n')

    # Initialize DQN
    DQN = DeepQNetwork(
        n_y=n_y,
        n_x=num_nodes,
        resources=resources,
        env=env,
        learning_rate=0.01,
        replace_target_iter=20,
        memory_size=20000,
        batch_size=256,
        reward_decay=0.6,
        epsilon_min=0.1,
        epsilon_greedy_decrement=1e-4,
        # load_path=load_path,
        # save_path=save_path
    )

    episodes = 1000
    rewards = []
    total_steps_counter = 0
    episodes_since_max = 0

    optimal_action_sequences = []
    overall_start = time.time()
    # DQN.epsilon = 0.5

    for episode in range(episodes):

        observation, done = env.reset()
        episode_reward = 0
        action_sequence = []
        start = time.time()
        train_time = 0

        while not done:
            # 1. Choose an action based on observation
            action = DQN.choose_action(observation)

            # check for random action
            if action == -1:
                # now choose between truly random action and a ratio action
                # r = random.random()
                action = env.random_action()
                # if r < 0.2 and episode < 0:
                #     action = env.random_action()
                # elif episode < 0:
                #     action = env.ratio_action()
                # else:
                #     action = env.random_action()

            # save the taken action
            action_sequence.append(action)

            # print('Chosen action', action)
            # 2. Take the chosen action in the environment
            observation_, reward, done = env.step(action, neg=False)
            # print(observation_, reward, done)
            # 3. Store transition
            DQN.store_transition(observation, action, reward, observation_)

            episode_reward += reward

            if total_steps_counter > 2000:
                # 4. Train
                s = time.time()
                DQN.learn()
                e = time.time()
                train_time += (e - s)

            if done:
                rewards.append(episode_reward)
                max_reward_so_far = np.amax(rewards)

                # if maximum reward so far, save the action sequence
                if episode_reward == max_reward_so_far:
                    optimal_action_sequences.append((action_sequence, episode_reward))
                    episodes_since_max = 0
                    # DQN.epsilon = 1

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", round(episode_reward, 2))
                print("Epsilon: ", round(DQN.epsilon, 2))
                print("Max reward so far: ", max_reward_so_far)

                end = time.time()
                print('Episode time:', end - start)
                start = time.time()

                break

            # Save observation
            observation = observation_

            # Increase total steps
            total_steps_counter += 1

            # if episode == 700:
            #     DQN.epsilon_min = .1
            #     DQN.epsilon = 0.5

        episodes_since_max += 1
        print('train time across episode', train_time)

    overall_end = time.time()

    # TEST Q-Learning
    DQN.epsilon = 0
    DQN.epsilon_min = 0
    observation, done = env.reset()
    final_reward = 0
    action_sequence = []
    while not done:
        action = DQN.choose_action(observation)
        action_sequence.append(action)
        observation_, reward, done = env.step(action, neg=False)

        final_reward += reward
        if done:
            rewards.append(final_reward)
            max_reward_so_far = np.amax(rewards)

            # if maximum reward so far, save the action sequence
            if final_reward == max_reward_so_far:
                optimal_action_sequences.append((action_sequence, final_reward))
                episodes_since_max = 0
            break

        # Save observation
        observation = observation_

    print()
    print('final epsilon=0 reward', final_reward)
    print()

    # TESTING
    # convert our best optimal action sequence to vector representation, test it for correctness
    opt = optimal_action_sequences[len(optimal_action_sequences) - 1][0]
    reward = optimal_action_sequences[len(optimal_action_sequences) - 1][1]

    print()
    # print('RL action sequence:')
    env.reset()
    true_r = 0
    for action in opt:
        # print('action index', action)
        _, r, d = env.step(action, debug=True)
        true_r += r

    # if we have a reasonable number of nodes, we can compute optimal
    if num_nodes < 24:
        dp_time = time.time()
        print("Optimal:", DP_optimal(G, [root], resources))
        dp_time_end = time.time()
        print('DP time:', dp_time_end - dp_time)

    print()
    print('random herustic', random_heuristic(G, [root], resources))
    print()
    print('Tree Heuristic:', simulate_tree_recovery(G, resources, root, clean=False))
    print()
    ratio_time_start = time.time()
    print("Ratio Heuristic", ratio_heuristic(G, [root], resources))
    ratio_time_end = time.time()
    print('Ratio time:', ratio_time_end - ratio_time_start)
    print()
    print('reward during training:', reward)
    print('RL method time (s): ', overall_end - overall_start)
    print()

    plot_bar_x(rewards, 'episode', 'reward_graph.png')
    with open(reward_save, 'w') as f:
        for item in rewards:
            f.write('%s\n' % item)


if __name__ == '__main__':
    main()
