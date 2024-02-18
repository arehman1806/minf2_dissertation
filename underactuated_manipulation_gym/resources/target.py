import pybullet as p
import pybullet_data
import os
import numpy as np
import yaml


class Target():
    def __init__(self, client, pose):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.rng = np.random.default_rng(seed=12345)
        config = self._load_config()
        self.spawn_config = config["spawn_config"]
        pos, orn = pose
        self.base_position = pos
        self.target = p.addUserDebugPoints(np.array([self.base_position]), pointColorsRGB=np.array([[0,1,0]]), pointSize=20, physicsClientId=self.client)
        self.target_text = p.addUserDebugText("Target", self.base_position, physicsClientId=self.client)
    
    """
    Returns the client and body ids
    """
    def get_ids(self):
        return self.client, self.target
    
    """
    Returns the state of the body
    """
    def get_base_position(self):
        return self.base_position
    
    """
    Resets the position of the body
    """
    def reset_position(self, position):
        if position is None:
            position = self._sample_pose()
        p.removeUserDebugItem(self.target, physicsClientId=self.client)
        p.removeUserDebugItem(self.target_text, physicsClientId=self.client)
        self.base_position = position
        self.target = p.addUserDebugPoints(np.array([self.base_position]), pointColorsRGB=np.array([[0,1,0]]), pointSize=20, physicsClientId=self.client)
        self.target_text = p.addUserDebugText("Target", self.base_position, physicsClientId=self.client)

    def _sample_pose(self):
        x = self.rng.uniform(self.spawn_config["min_x"], self.spawn_config["max_x"])
        y = self.rng.uniform(self.spawn_config["min_y"], self.spawn_config["max_y"])
        z = 0.0
        return [x, y, z]
    
    def _load_config(self):
        current_file = __file__
        current_directory = os.path.dirname(current_file)
        with open(f"{current_directory}/target_config.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setGravity(0,0,-10)
    p.setRealTimeSimulation(0)
    # p.setTimeStep(1./500)
    target = Target(0, "target", ([5,5,0.0], [0,0,0,1]))
    for _ in range(10000000000000000):
        p.stepSimulation()