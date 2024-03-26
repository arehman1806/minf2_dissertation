import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
import time
import numpy as np
import pybullet as p

debug = True
from rule_based_policies.push_delta import PushDeltaPolicy

env = gym.make("queenie_gym_envs/PushDeltaEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_delta_environment.yaml")
env = DummyVecEnv([lambda: env])
obs = env.reset()
model = PushDeltaPolicy(env)
if debug:
    p.connect(p.SHARED_MEMORY)
    visualShapeId = p.createVisualShape(shapeType=p.GEOM_SPHERE, rgbaColor=[1, 1, 1, 1], radius=0.01)
    collisionShapeId = -1  #p.createCollisionShape(shapeType=p.GEOM_MESH, fileName="duck_vhacd.obj", collisionFramePosition=shift,meshScale=meshScale)
    start_stop_button = p.addUserDebugParameter("start/stop", 1, 0, 0)
    last_start_stop = -1
env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if debug:
        p.createMultiBody(baseMass=0,
                           baseCollisionShapeIndex=collisionShapeId,
                           baseVisualShapeIndex=visualShapeId,
                           basePosition=p.getBasePositionAndOrientation(2)[0],
                           useMaximalCoordinates=True)
        start_stop = p.readUserDebugParameter(start_stop_button)
        if start_stop != last_start_stop:
            last_start_stop = start_stop + 1
            while last_start_stop != start_stop:
                start_stop = p.readUserDebugParameter(start_stop_button)
                time.sleep(0.01)
        if i == 499:
            input("Press Enter to continue...")
    if done:
        if debug:
            input("Press Enter to continue...")
            break
        print("done at step ", i)
        # env.reset()
