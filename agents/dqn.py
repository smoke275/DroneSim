import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import MlpPolicy

from agents.agent import Agent

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            ep_rewards = [info["r"] for info in self.model.ep_info_buffer]
            mean_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else -float("inf")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path)
        return True

class DQNAgent(Agent):
    def __init__(self, env, config, policy_path=None, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        
        self.model = DQN(
            policy="MlpPolicy", 
            env=env,
            learning_rate=1e-2,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=256,
            gamma=0.99,
            target_update_interval=10,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=f"runs/dqn/tensorboard/"
        )

        if policy_path is not None and os.path.exists(policy_path):
            self.model = DQN.load(policy_path)
            self.model.set_env(env)
            print(f"Policy loaded from {policy_path}")

    def predict(self, observation):
        """For inference, return the best action for the given observation."""
        action, _state = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, num_timesteps=100000, policy_path=None, log_path=None):
        """Train the DQN agent"""
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=policy_path)
        self.model.learn(total_timesteps=num_timesteps, callback=callback)
        
        # Save final model
        self.model.save(policy_path)
        print(f"Final policy saved to {policy_path}")
