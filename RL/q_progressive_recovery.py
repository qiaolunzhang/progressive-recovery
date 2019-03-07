from deep_q_network import DeepQNetwork
from rl_environment import environment
import networkx as nx
from tree_recovery import r_tree, get_root, DP_optimal, plot_graph
import numpy as np
import random
import itertools

# Load checkpoint
load_path = "model/weights.ckpt"
save_path = "model/weights.ckpt"

# random graph
num_nodes = 15
resources = 1
G = r_tree(num_nodes)

plot_graph(G, get_root(G), 'rl_graph.png')
env = environment(G, [get_root(G)], resources)
n_y = len(list(itertools.permutations(range(num_nodes), 2)))

# Initialize DQN
DQN = DeepQNetwork(
    n_y=n_y,
    n_x=num_nodes,
    resources=resources,
    learning_rate=0.01,
    replace_target_iter=100,
    memory_size=2000,
    batch_size=64,
    reward_decay=1,
    epsilon_min=0.1,
    epsilon_greedy_decrement=0.001,
    # load_path=load_path,
    # save_path=save_path
)

print("Optimal:", DP_optimal(G, [get_root(G)], resources))

EPISODES = 1500
rewards = []
total_steps_counter = 0

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
            #action = env.random_action()
            action = random.randint(0, n_y - 1)

        # save the taken action
        action_sequence.append(action)

        #print('Chosen action', action)
        # 2. Take the chosen action in the environment
        observation_, reward, done = env.step(action)
        #print(observation_, reward, done)

        # 3. Store transition
        DQN.store_transition(observation, action, reward, observation_)

        episode_reward += reward

        if total_steps_counter > 100:
            # 4. Train
            DQN.learn()

        if done:
            rewards.append(episode_reward)
            max_reward_so_far = np.amax(rewards)

            # if maximum reward so far, save the action sequence
            if episode_reward == max_reward_so_far:
                optimal_action_sequences.append((action_sequence, episode_reward))

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


print("Optimal:", DP_optimal(G, [get_root(G)], resources))


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
