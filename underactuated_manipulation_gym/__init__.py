

from gymnasium.envs.registration import register

register(
     id="queenie_gym_envs/DifferentialDriveEnv-v0",
     entry_point="underactuated_manipulation_gym.envs:DifferentialDriveEnv",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/GraspEnvironment-v0",
     entry_point="underactuated_manipulation_gym.envs:GraspEnvironment",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/GraspEnvironment-v1",
     entry_point="underactuated_manipulation_gym.envs:GraspEnvironment1",
     max_episode_steps=100,
)

register(
     id="queenie_gym_envs/PushEnvironment-v0",
     entry_point="underactuated_manipulation_gym.envs:PushEnvironment",
)

register(
     id="queenie_gym_envs/MetaEnvironment-v0",
     entry_point="underactuated_manipulation_gym.envs:MetaEnvironment",
)