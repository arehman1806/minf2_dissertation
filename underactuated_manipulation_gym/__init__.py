

from gymnasium.envs.registration import register

register(
     id="queenie_gym_envs/DifferentialDriveEnv-v0",
     entry_point="underactuated_manipulation_gym.envs:DifferentialDriveEnv",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/RandomURDFsSOEnvironment-v0",
     entry_point="underactuated_manipulation_gym.envs:RandomURDFsSOEnvironment",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/RandomURDFsSOEnvironment-v1",
     entry_point="underactuated_manipulation_gym.envs:RandomURDFsSOEnvironment1",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/PushEnvironment-v0",
     entry_point="underactuated_manipulation_gym.envs:PushEnvironment",
)