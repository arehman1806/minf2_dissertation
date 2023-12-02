import underactuated_manipulation_gym
import gymnasium as gym
import time
import numpy as np
import pybullet as p
import cv2
env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/simple_manipulation.yaml")
env.reset()
j = 0
interface_neck_y = p.addUserDebugParameter("neck_y", -1.57, 1.57, 0)
interface_neck_x = p.addUserDebugParameter("neck_x", -1.57, 1.57, 0)
interface_gripper_position = p.addUserDebugParameter("gripper position", -0.1, 0.1, 1)
interface_v = p.addUserDebugParameter("v", -2, 2, 0)
interface_w_angular = p.addUserDebugParameter("w", -2, 2, 0)
for i in range(1000000):
    v = p.readUserDebugParameter(interface_v)
    w_angular = p.readUserDebugParameter(interface_w_angular)
    neck_y = p.readUserDebugParameter(interface_neck_y)
    neck_x = p.readUserDebugParameter(interface_neck_x)
    gripper_pos = p.readUserDebugParameter(interface_gripper_position)
    action = np.array([v, w_angular, neck_y, neck_x, gripper_pos])
    action = np.array([v, w_angular, neck_y, neck_x])
    # action = env.action_space.sample()
    # action = np.array([0.0, 0.0, 0.0, 0.0, 1])
    obs, reward, done, _, _ = env.step(action)
    # img_to_save = obs["image_obs"]
    # img_to_save = np.transpose(img_to_save, (1, 2, 0))
    # cv2.imwrite("test.png", img_to_save)
    # print(obs["image_obs"].shape)
    # if done:
    #     print(f"done {j}")
    #     env.reset()
    #     j += 1
    # time.sleep(1)