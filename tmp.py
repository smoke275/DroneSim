import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
# Using make_vec_env for potential parallelization, though n_envs=1 here
env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=1)

# Instantiate the agent
# DQN is suitable for environments with discrete action spaces like CartPole
model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=1e-4,
            buffer_size=50000,       # size of the replay buffer
            learning_starts=1000,    # how many steps of the model to collect transitions for before learning starts
            batch_size=32,           # size of a batched sampled from replay buffer for training
            tau=1.0,                 # the soft update coefficient ("Polyak update", 1.0 means hard update)
            gamma=0.99,              # the discount factor
            train_freq=4,            # update the model every `train_freq` steps
            gradient_steps=1,        # how many gradient steps to do after each update
            target_update_interval=250, # update the target network every `target_update_interval` environment steps
            exploration_fraction=0.1,   # fraction of entire training period over which the exploration rate is reduced
            exploration_initial_eps=1.0,# initial value of random action probability
            exploration_final_eps=0.05, # final value of random action probability
            tensorboard_log="./dqn_cartpole_tensorboard/")

# Train the agent
print("Starting training...")
total_timesteps = 50000
model.learn(total_timesteps=total_timesteps, log_interval=10) # Log every 10 episodes
print("Training finished.")

# Save the agent
model.save("dqn_cartpole")
print(f"Model saved to dqn_cartpole.zip")

# Evaluate the trained agent
print("Evaluating the trained agent...")
# Use a separate environment for evaluation
eval_env = gym.make(env_id)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Evaluation results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

# Clean up resources
env.close()
eval_env.close()

# To load the model later:
# del model # remove to demonstrate loading
# model = DQN.load("dqn_cartpole", env=env)
# print("Model loaded.")
# You can then continue training or use it for inference