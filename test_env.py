import underactuated_manipulation_gym
import gymnasium as gym
import time
env = gym.make("queenie_gym_envs/DifferentialDriveEnv-v0")
env.reset()
j = 0
for i in range(1000000):
    print(f"starting sleep")
    time.sleep(1)
    print(f"step {i}")
    obs, reward, done, _, _ = env.step(env.action_space.sample())
    if done:
        print(f"done {j}")
        env.reset()
        j += 1
    # time.sleep(1)