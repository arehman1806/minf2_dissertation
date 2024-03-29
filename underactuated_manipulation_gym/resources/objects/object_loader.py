from pathlib import Path
import os
import numpy as np
import yaml
import pybullet as p
import pybullet_data

# from man_object import ObjectMan
from underactuated_manipulation_gym.resources.objects.man_object import ObjectMan


class ObjectLoader:

    def __init__(self, client, object_class, specific_objects=[], num_objects=-1, global_scale=4) -> None:
        self.client = client
        self.rng = np.random.default_rng(seed=12345)
        config = self._load_config()
        self.spawn_config = config["spawn_config"]
        if "dynamics" in config.keys():
            self.dynamics = config["dynamics"]
        else:
            self.dynamics = None
        object_class = config["object_class"]
        self.global_scale = global_scale
        if object_class == "random_urdfs":
            self.object_paths = sorted([str(file) for file in Path(f"{pybullet_data.getDataPath()}/{object_class}/").rglob('*.urdf') if file.is_file()])
            if len(specific_objects) != 0:
                self.object_paths = [self.object_paths[i] for i in specific_objects]

            elif num_objects != -1:
                self.object_paths = self.object_paths[0:num_objects]
        else:
            self.object_paths = self._load_object_paths(object_class)
        if len(self.object_paths) == 0:
            raise Exception(f"No objects found for {object_class}")
        
        self.current_object = None

    def change_object(self, pose=None):
        if self.current_object is not None:
            self._remove_object()
        self.current_object = self._spawn_object(pose)
        print(f"current object: {self.current_object.get_ids()}")
        if self.dynamics is not None:
            p.changeDynamics(self.current_object.get_ids()[1], -1, spinningFriction=self.dynamics["spinning_friction"])
        return self.current_object

    def empty_scene(self):
        return self._remove_object()
    
    def get_current_object(self):
        return self.current_object
    
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
        with open(f"{current_directory}/manipulation_object.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config


if __name__ == "__main__":
    ol = ObjectLoader(0, "objects_cuboid")
    print(ol._sample_pose())