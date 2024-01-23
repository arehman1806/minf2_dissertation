import pybullet as p
import pybullet_data
import os


class ObjectMan():
    def __init__(self, client, object_path, pose, scale=1):
        self.client = client
        pos, orn = pose
        self.manipulation_object = p.loadURDF(object_path, pos, orn, globalScaling=scale, physicsClientId=self.client)
    
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.manipulation_object
    

    def get_state(self):
        return p.getBasePositionAndOrientation(self.manipulation_object, physicsClientId=self.client)
    
    def get_base_pose(self):
        return p.getBasePositionAndOrientation(self.manipulation_object, physicsClientId=self.client)
    
    def remove_from_sim(self):
        p.removeBody(self.manipulation_object, physicsClientId=self.client)


if __name__ == "__main__":
    pass