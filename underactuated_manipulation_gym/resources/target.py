import pybullet as p
import pybullet_data
import os
import numpy as np


class Target():
    def __init__(self, client, pose):
        self.client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pos, orn = pose
        self.base_position = pos
        self.target = p.addUserDebugPoints(np.array([self.base_position]), pointColorsRGB=np.array([[1,1,0]]), pointSize=20, physicsClientId=self.client)
    
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
        self.base_position = position
        self.target = p.addUserDebugPoints(np.array([self.base_position]), pointColorsRGB=np.array([[1,1,0]]), pointSize=20, physicsClientId=self.client)

    def _sample_pose(self):
        return [3, 3, 0]

if __name__ == "__main__":
    p.connect(p.GUI)
    p.setGravity(0,0,-10)
    p.setRealTimeSimulation(0)
    # p.setTimeStep(1./500)
    target = Target(0, "target", ([5,5,0.0], [0,0,0,1]))
    for _ in range(10000000000000000):
        p.stepSimulation()