import numpy as np


from .sensor import Sensor
from .camera import Camera_Sensor

class PointCloudSensor(Sensor):

    def __init__(self, client, robot, sensor_name, sensor_params, robot_params, camera: Camera_Sensor):
        super().__init__(robot, sensor_name, sensor_params, robot_params)

        self.camera = camera
        self.resolution = self.camera.resolution
        self.client = client


    def get_observation(self):
        return self._get_point_cloud(self.camera.get_current_depth_image())

    def _get_point_cloud(self, depth):
        # Get the point cloud from the depth image
        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.camera.proj_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.camera.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.resolution, -1:1:2 / self.resolution]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points