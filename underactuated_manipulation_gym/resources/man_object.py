import pybullet as p
import pybullet_data
import os


class ObjectMan():
    def __init__(self, client, object_path, pose):
        self.client = client
        pos, orn = pose
        self.manipulation_object = p.loadURDF(object_path, pos, orn, physicsClientId=self.client)
    
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.manipulation_object
    
    def remove_from_sim(self):
        p.removeBody(self.manipulation_object, physicsClientId=self.client)


if __name__ == "__main__":
    pass