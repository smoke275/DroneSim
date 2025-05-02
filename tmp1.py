import gymnasium as gym
from stable_baselines3 import DQN
import time

# Define the environment ID and the path to the saved model
env_id = "CartPole-v1"
model_path = "dqn_cartpole.zip"

# Load the trained agent
# Note: You might need to pass the same environment instance or configuration
# if the environment requires specific parameters upon loading.
# For CartPole-v1, loading without the env argument often works,
# but explicitly passing it is safer if environment parameters were customized.
# env = gym.make(env_id) # Optional: Create env instance first
# model = DQN.load(model_path, env=env)
model = DQN.load(model_path)
print(f"Model loaded from {model_path}")

# Create the environment for inference
# Add render_mode='human' to visualize the agent's performance
env = gym.make(env_id, render_mode='human')

print("Starting inference...")
obs, info = env.reset()
terminated = False
truncated = False
episode_reward = 0
episode_length = 0

while not terminated and not truncated:
    # Use deterministic=True for the model to choose the best action
    # action, _states = model.predict(obs, deterministic=True)
    action = env.action_space.sample()

    # Perform the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Accumulate reward and step count
    episode_reward += reward
    episode_length += 1

    # Render the environment
    env.render()

    # Optional: Add a small delay to make visualization easier to follow
    time.sleep(0.02)

    # Check if the episode has ended
    if terminated or truncated:
        print(f"Episode finished. Reward: {episode_reward}, Length: {episode_length}")
        # Optionally reset and run more episodes
        # obs, info = env.reset()
        # terminated = False
        # truncated = False
        # episode_reward = 0
        # episode_length = 0

# Clean up resources
env.close()
print("Inference finished and environment closed.")

