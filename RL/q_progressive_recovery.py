from deep_q_network import DeepQNetwork
from rl_environment import environment
import networkx as nx
from graph_helper import r_graph, r_2d_graph, r_tree, get_root, DP_optimal, plot_graph, simulate_tree_recovery, \
    plot_bar_x, read_gml, adv_graph, read_gml_adversarial, gnp_adversarial
import numpy as np
from ratio_heuristic import ratio_heuristic
from random_heuristic import random_heuristic
import time
import random
import tensorflow as tf


def generate_graph(nodes=20, utils=[1, 4], demands=[1, 2], load_dir=None, type='random_tree', seed=None):
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
            graph = r_graph(nodes, 0.2, utils, demands, seed)
        save = 'experiments/{0}_rgraph.txt'.format(nodes)
        real_node_num = nodes

    # Generate random nodes x nodes 2-d grid graph
    elif type == 'grid':
        if load_dir:
            graph = nx.read_gpickle('experiments/{0}x{0}.gpickle'.format(nodes))
        else:
            graph = r_2d_graph(nodes, nodes, utils, demands)
        save = 'experiments/{0}x{0}.txt'.format(nodes)
        real_node_num = nodes ** 2

    # Generate adversarial examples for the ratio heuristic
    elif type == 'adversarial':
        graph = adv_graph(nodes, utils, demands)
        save = 'experiments/{0}_adv_graph.txt'.format(nodes)
        real_node_num = nodes

    # Read a normal gml file and randomly pick utils, demands
    elif type == 'gml':
        # WARNING: Turn off fix_nodes_around_adv for normal use. This fixes the
        # nodes adjacent to vertex(num_nodes - 2) to be bad for the ratio heuristic.
        # For use in conjunction with 'gml_adversarial'.
        graph = read_gml(load_dir, utils, demands, fix_nodes_around_adv=False)
        save = 'experiments/{0}.txt'.format('gml')
        real_node_num = len(graph)

    # Read a gml file and randomly pick utils, demands but also
    # embed an adversarial example somewhere in the graph.
    elif type == 'gml_adversarial':
        graph = read_gml_adversarial(load_dir, utils, demands)
        save = 'experiments/{0}_adv_graph.txt'.format('gml')
        real_node_num = len(graph)

    elif type == 'gnp_adversarial':
        graph = gnp_adversarial(nodes, utils, demands)
        save = 'experiments/{0}_gnp_adv_graph.txt'.format(nodes)
        real_node_num = nodes

    else:
        raise NotImplementedError

    # Real number of nodes may be different from input node num (in the case of grid graph)
    return graph, save, real_node_num


def runner(node_num):
    # Load checkpoint
    load_path = "weights/weights.ckpt"
    save_path = "weights/weights.ckpt"

    # set seed
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph for training...
    resources = 1
    # G, reward_save, num_nodes = generate_graph(nodes=node_num, type='gnp_adversarial')
    # G, reward_save, num_nodes = generate_graph(load_dir='../gml/ibm.gml', type='gml')
    G, reward_save, num_nodes = generate_graph(nodes=node_num, type='random_graph', seed=42)

    # Pick an arbitrary node to be the root
    root = 0
    # Try plotting. If on ssh, don't bother since there are some necessary plt.draw() commands
    # to plot a networkx graph.
    try:
        plot_graph(G, root, 'rl_graph.png')
    except:
        print('No display')

    # We may want to include the graph laplacian in the observation space
    # Graph laplacian is D - A
    # laplacian_matrix = nx.laplacian_matrix(G).toarray()
    # flat_laplacian = laplacian_matrix.flatten()

    # Build the learning environment
    env = environment(G, [root], resources)
    print('num_edges:', G.number_of_edges())
    print("Ratio Heuristic", ratio_heuristic(G, [root], resources), '\n')

    # Our observation space
    n_y = len(env.actions_permutations)

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
        epsilon_greedy_decrement=5e-5,
        # load_path=load_path,
        # save_path=save_path,
        # laplacian=flat_laplacian,
        inner_act_func='leaky_relu',
        output_act_func='leaky_relu'
    )

    episodes = 600
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
                # action = env.random_action()
                # now choose between truly random action and a ratio action
                r = random.random()
                if r < 0.6:
                    action = env.random_action()
                else:
                    action = env.ratio_action()

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

    print('final epsilon=0 reward', final_reward, '\n')

    # TESTING
    # convert our 'best' optimal action sequence to the vector representation, test it for correctness
    opt = optimal_action_sequences[len(optimal_action_sequences) - 1][0]
    reward = optimal_action_sequences[len(optimal_action_sequences) - 1][1]

    print()
    # print('RL action sequence:')
    env.reset()
    true_r = 0
    for action in opt:
        # print('action index', action)
        # debug will print the action at each step as a vector
        _, r, d = env.step(action, debug=True)
        true_r += r

    results = []
    # if we have a reasonable number of nodes (< 24), we can compute optimal using DP
    if num_nodes < 24:
        dp_time = time.time()
        results.append(DP_optimal(G, [root], resources))
        print('DP Opt: ', results[0])
        dp_time_end = time.time()
        results.append(dp_time_end - dp_time)
        print('DP time: ', results[1])
    else:
        results.append('n/a')
        results.append('n/a')

    print('\n Random Heuristic', random_heuristic(G, [root], resources), '\n')
    results.append(random_heuristic(G, [root], resources))

    # Only works on trees
    # print('\n Tree Heuristic:', simulate_tree_recovery(G, resources, root, clean=False), '\n')

    ratio_time_start = time.time()
    print('\n Ratio Heuristic', ratio_heuristic(G, [root], resources))
    ratio_time_end = time.time()
    print('Ratio time:', ratio_time_end - ratio_time_start)
    results.append(ratio_heuristic(G, [root], resources))
    results.append(ratio_time_end - ratio_time_start)

    print('\n reward during training:', reward)
    results.append(reward)
    print('RL method time (s): ', overall_end - overall_start, '\n')
    results.append(overall_end - overall_start)

    plot_bar_x(rewards, 'episode', 'reward_graph.png')
    with open(reward_save, 'w') as f:
        for item in rewards:
            f.write('%s\n' % item)

    return results


def main():
    all_res = []
    for node_num in range(20, 21):
        all_res.append(runner(node_num))
        tf.reset_default_graph()

    # print all results formatted as csv
    for node_num in all_res:
        try:
            print(node_num[0][0],',', node_num[1], ',', node_num[2],',',  node_num[3], ',', node_num[4],',',  node_num[5],',',  node_num[6])
        except:
            print(node_num[0],',',  node_num[1], ',', node_num[2],',',  node_num[3],',',  node_num[4])


if __name__ == '__main__':
    main()
