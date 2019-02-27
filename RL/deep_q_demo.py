"""
Deep Q Network
CartPole-v0
"""

import gym
from deep_q_network import DeepQNetwork
from rl_env import env

# Load checkpoint
load_path = "weights/CartPole-v0.ckpt"
save_path = "weights/CartPole-v0-2.ckpt"

# Initialize DQN
DQN = DeepQNetwork(
    n_y=env.action_space.n,
    n_x=env.observation_space.shape[0],
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


RENDER_ENV = True
EPISODES = 10000
rewards = []
RENDER_REWARD_MIN = 800
total_steps_counter = 0


for episode in range(EPISODES):

    observation = env.reset()
    episode_reward = 0

    for _ in range(200):
        if RENDER_ENV: env.render()

        # 1. Choose an action based on observation
        action = DQN.choose_action(observation)

        # 2. Take the chosen action in the environment
        observation_, reward, done, info = env.step(action)

        # x, x_dot, theta, theta_dot = observation_
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # reward = r1 + r2
        #
        # print(reward)

        # 3. Store transition
        DQN.store_transition(observation, action, reward, observation_)

        episode_reward += reward

        if total_steps_counter > 1000:
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

            # Render env if we get to rewards minimum
            if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True

            break

        # Save observation
        observation = observation_

        # Increase total steps
        total_steps_counter += 1

# DQN.plot_cost()
