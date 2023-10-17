import pybullet as p
import pybullet_data
import os


class Plane():
    def __init__(self, client):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=client)

if __name__ == "__main__":
    pass