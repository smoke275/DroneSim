import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import yaml

# Load config
with open("config/ral.yaml", "r") as file:
    env_config = yaml.safe_load(file)

def make_env():
    import envs  # This will register the environment
    return gym.make('LMDEnv-v0', config=env_config)

if __name__ == "__main__":
    # Initialize wandb
    run = wandb.init(
        project="lmd-training",
        config=env_config,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    # Create vectorized environment
    env = make_vec_env(make_env, n_envs=4)

    # Initialize the agent
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{run.id}",
        name_prefix="lmd_model"
    )

    # Train the agent
    model.learn(
        total_timesteps=1000000,
        callback=[
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            checkpoint_callback
        ],
    )

    # Save the final model
    model.save(f"models/{run.id}/final_model")
    
    # Close environment and wandb run
    env.close()
    run.finish()
