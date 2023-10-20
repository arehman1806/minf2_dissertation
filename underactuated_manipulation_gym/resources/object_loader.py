from pathlib import Path
import os
import numpy as np
import pybullet as p

from underactuated_manipulation_gym.resources.man_object import ObjectMan


class ObjectLoader:

    def __init__(self, client, object_class) -> None:
        self.client = client
        self.object_paths = self._load_object_paths(object_class)
        print(self.object_paths)
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
        return ObjectMan(self.client, self.rng.choice(self.object_paths), pose)

    def _remove_object(self):
        self.current_object.remove_from_sim()
        self.current_object = None
        return 1

    def _sample_pose(self):
        x = self.rng.uniform(-0.5, 0.5)
        y = self.rng.uniform(-0.5, 0.5)
        z = 0.1
        pos = [x, y, z]
        orn = p.getQuaternionFromEuler([0, 0, self.rng.uniform(0, 2 * np.pi)])
        return pos, orn
    
    def _load_object_paths(self, object_class):
        current_file = __file__
        current_directory = os.path.dirname(current_file)
        return [str(file) for file in Path(f"{current_directory}/urdfs/{object_class}/").rglob('*') if file.is_file()]

if __name__ == "__main__":
    ol = ObjectLoader(0, "objects_cuboid")