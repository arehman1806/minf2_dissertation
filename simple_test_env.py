import gymnasium as gym
# from stable_baselines3 import SAC
import underactuated_manipulation_gym
import time
import numpy as np

# IMPORTANT. edit this before every run:
tb_log_name = "first_run"

env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/simple_manipulation.yaml")

for i in range(100):
    env.reset()
    for j in range(1000):
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        if done:
            print(f"done {j}")
            break
        # time.sleep(1)

# env.reset()
# model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=500000, tensorboard_log="./logs/simple_multi_object_pickup_agent")
# model.learn(total_timesteps=500000, log_interval=10, tb_log_name=tb_log_name)
# model.save("sac_queenie_multi_object")
# model = SAC.load("sac_queenie", env=env)
# vec_env = model.get_env()
# obs= vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
