

from gymnasium.envs.registration import register

register(
     id="queenie_gym_envs/DifferentialDriveEnv-v0",
     entry_point="underactuated_manipulation_gym.envs:DifferentialDriveEnv",
     max_episode_steps=100,
)

