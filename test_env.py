import underactuated_manipulation_gym
import gymnasium as gym
import time
env = gym.make("queenie_gym_envs/DifferentialDriveEnv-v0")
env.reset()
for i in range(1000000):
    obs, reward, done, _, _ = env.step(env.action_space.sample())
    if done:
        print("done")
        env.reset()
    # time.sleep(1)