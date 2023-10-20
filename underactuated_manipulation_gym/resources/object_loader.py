import pybullet as p


class ObjectLoader:

    def __init__(self, client) -> None:
        self.client = client
        self.objects = self._load_object_paths()
        self.current_object = None

    def change_object(self, pose=None):
        pass

    def empty_scene(self):
        pass
    
    def _spawn_object(self, pose):
        pass

    def _remove_object(self):
        pass
