import pybullet as p
import pybullet_data
import os


class ObjectMan():
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "urdfs/cuboid.urdf")
        self.robot = p.loadURDF(f_name, [1.5, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.client)


if __name__ == "__main__":
    pass