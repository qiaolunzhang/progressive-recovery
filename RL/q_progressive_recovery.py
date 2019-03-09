from deep_q_network import DeepQNetwork
from rl_environment import environment
import networkx as nx
from tree_recovery import r_graph, r_tree, get_root, DP_optimal, plot_graph, simulate_tree_recovery, plot_bar_x
import numpy as np
import random
import itertools
from ratio_heuristic import ratio_heuristic

# Load checkpoint
load_path = "model/weights.ckpt"
save_path = "model/weights.ckpt"

# random graph
num_nodes = 22
resources = 1
#G = r_tree(num_nodes)
G = r_graph(num_nodes, 0.2)

try:
    plot_graph(G, get_root(G), 'rl_graph.png')
except:
    print('No display')

env = environment(G, [get_root(G)], resources)
n_y = len(env.actions_permutations)

# Initialize DQN
DQN = DeepQNetwork(
    n_y=n_y,
    n_x=num_nodes,
    resources=resources,
    env=env,
    learning_rate=0.1,
    replace_target_iter=100,
    memory_size=20000,
    batch_size=64,
    reward_decay=0.4,
    epsilon_min=0.2,
    epsilon_greedy_decrement=0.00001,
    # load_path=load_path,
    # save_path=save_path
)

EPISODES = 10000
rewards = []
total_steps_counter = 0
episodes_since_max = 0

optimal_action_sequences = []

for episode in range(EPISODES):

    observation, done = env.reset()
    episode_reward = 0
    action_sequence = []

    while not done:
        # 1. Choose an action based on observation
        action = DQN.choose_action(observation)

        # check for random action
        if action == -1:
            action = env.random_action()

        # save the taken action
        action_sequence.append(action)

        #print('Chosen action', action)
        # 2. Take the chosen action in the environment
        observation_, reward, done = env.step(action)
        #print(observation_, reward, done)

        # 3. Store transition
        DQN.store_transition(observation, action, reward, observation_)

        episode_reward += reward

        if total_steps_counter > 1000:
            # 4. Train
            DQN.learn()

        if done:
            rewards.append(episode_reward)
            max_reward_so_far = np.amax(rewards)

            # if maximum reward so far, save the action sequence
            if episode_reward == max_reward_so_far:
                optimal_action_sequences.append((action_sequence, episode_reward))
                episodes_since_max = 0

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", round(episode_reward, 2))
            print("Epsilon: ", round(DQN.epsilon,2))
            print("Max reward so far: ", max_reward_so_far)

            break

        # Save observation
        observation = observation_

        # Increase total steps
        total_steps_counter += 1

    episodes_since_max += 1
    if episodes_since_max > 2000:
        break


print("Optimal:", DP_optimal(G, [get_root(G)], resources))
#print('Tree Heuristic:', simulate_tree_recovery(G, resources, get_root(G), clean=False))
print("Ratio Heuristic", ratio_heuristic(G, [get_root(G)], resources))

# TESTING
# convert our best optimal action sequence to vector representation, test it for correctness
opt = optimal_action_sequences[len(optimal_action_sequences) - 1][0]
reward = optimal_action_sequences[len(optimal_action_sequences) - 1][1]

print()
print('Optimal RL action sequence:')
env.reset()
true_r = 0
for action in opt:
    print('action index', action)
    _, r, d = env.step(action, debug=True)
    true_r += r

print('reward during training:', reward, 'reward during testing:', true_r)
plot_bar_x(rewards, 'episode', 'reward_graph.png')
