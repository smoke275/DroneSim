from gymnasium.envs.registration import register
from envs.ral import LMDEnv

register(
    id='LMDEnv-v0',
    entry_point=LMDEnv,
    max_episode_steps=1000,  # or any suitable number
)
