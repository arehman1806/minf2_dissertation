import pybullet as p
import pybullet_data
import os


class ObjectMan():
    def __init__(self, client, pose):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "urdfs/cuboid.urdf")
        pos, orn = pose
        self.manipulation_object = p.loadURDF(f_name, pos, orn, physicsClientId=self.client)
    
    def remove_from_sim(self):
        p.removeBody(self.manipulation_object, physicsClientId=self.client)


if __name__ == "__main__":
    pass