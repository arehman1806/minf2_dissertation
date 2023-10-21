from pathlib import Path
import os
import numpy as np
import yaml
import pybullet as p
import pybullet_data

# from man_object import ObjectMan
from underactuated_manipulation_gym.resources.man_object import ObjectMan


class ObjectLoader:

    def __init__(self, client, object_class) -> None:
        self.client = client
        self.spawn_config = self._load_config()["spawn_config"]
        self.global_scale = 1
        if object_class == "random_urdfs":
            self.object_paths = [str(file) for file in Path(f"{pybullet_data.getDataPath()}/{object_class}/").rglob('*.urdf') if file.is_file()]
            self.global_scale = 4
        else:
            self.object_paths = self._load_object_paths(object_class)
        if len(self.object_paths) == 0:
            raise Exception(f"No objects found for {object_class}")
        
        self.current_object = None
        self.rng = np.random.default_rng()

    def change_object(self, pose=None):
        if self.current_object is not None:
            self._remove_object()
        self.current_object = self._spawn_object(pose)
        return self.current_object

    def empty_scene(self):
        return self._remove_object()
    
    def _spawn_object(self, pose):
        if pose is None:
            pose = self._sample_pose()
        return ObjectMan(self.client, self.rng.choice(self.object_paths), pose, self.global_scale)

    def _remove_object(self):
        self.current_object.remove_from_sim()
        self.current_object = None
        return 1

    def _sample_pose(self):
        x = self.rng.uniform(self.spawn_config["min_x"], self.spawn_config["max_x"])
        y = self.rng.uniform(self.spawn_config["min_y"], self.spawn_config["max_y"])
        z = 0.1
        pos = [x, y, z]
        orn = p.getQuaternionFromEuler([0, 0, self.rng.uniform(self.spawn_config["min_yaw"], self.spawn_config["max_yaw"])])
        return pos, orn
    
    def _load_object_paths(self, object_class):
        current_file = __file__
        current_directory = os.path.dirname(current_file)
        return [str(file) for file in Path(f"{current_directory}/urdfs/{object_class}/").rglob('*') if file.is_file()]
    
    def _load_config(self):
        current_file = __file__
        current_directory = os.path.dirname(current_file)
        with open(f"{current_directory}/config/manipulation_object.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config


if __name__ == "__main__":
    ol = ObjectLoader(0, "objects_cuboid")
    print(ol._sample_pose())