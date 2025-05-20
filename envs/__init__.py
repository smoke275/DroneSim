from gymnasium.envs.registration import register
from envs.ral2 import MultiAgentLMDEnv
from envs.ral import LMDEnv

register(
    id='MultiAgentLMDEnv-v0',
    entry_point=MultiAgentLMDEnv,
    max_episode_steps=10000,  # or any suitable number
)

register(
    id='LMDEnv-v0',
    entry_point=LMDEnv,
    max_episode_steps=10000,  # or any suitable number
)
