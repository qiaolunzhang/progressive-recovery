from deep_q_network import DeepQNetwork
from rl_env import environment
import networkx as nx
from tree_recovery import r_tree, get_root, DP_optimal, plot_graph
import numpy as np

# Load checkpoint
load_path = "model/weights.ckpt"
save_path = "model/weights.ckpt"

num_nodes = 7
resources = 1

G = r_tree(num_nodes)
plot_graph(G, get_root(G), 'rl_graph.png')
env = environment(G, [get_root(G)], resources)

# Initialize DQN
DQN = DeepQNetwork(
    n_y=num_nodes,
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

EPISODES = 1000
rewards = []
total_steps_counter = 0

for episode in range(EPISODES):

    observation, done = env.reset()
    episode_reward = 0

    while not done:
        # 1. Choose an action based on observation
        action = DQN.choose_action(observation)
        print('Chosen action', action)
        # 2. Take the chosen action in the environment
        observation_, reward, done = env.step(action)
        print(observation_, reward, done)

        # 3. Store transition
        DQN.store_transition(observation, action, reward, observation_)

        episode_reward += reward

        if total_steps_counter > 100:
            # 4. Train
            DQN.learn()

        if done:
            rewards.append(episode_reward)
            max_reward_so_far = np.amax(rewards)

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