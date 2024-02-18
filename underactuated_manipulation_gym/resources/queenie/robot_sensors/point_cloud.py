import numpy as np


from .sensor import Sensor
from .camera import Camera_Sensor
import open3d as o3d

class PointCloudSensor(Sensor):

    def __init__(self, client, robot, sensor_name, sensor_params, robot_params, camera: Camera_Sensor, object_handler=None):
        super().__init__(robot, sensor_name, sensor_params, robot_params)

        self.camera = camera
        self.resolution = self.camera.resolution
        self.client = client
        self.object_handler = object_handler


    def get_observation(self):
        return self._get_point_cloud(self.camera.get_current_depth_image())

    def _get_point_cloud(self, depth):
        semantic_map = self.camera.current_semantic_image
        current_object_id = self.object_handler.get_current_object().get_ids()[1]
        mask = semantic_map == current_object_id
        depth = depth * mask
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
        # pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1
        pixels = pixels[pixels[:, 2] > -1]

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        actual_points = len(points)
        required_points = depth.shape[0] * depth.shape[1]
        if actual_points < required_points:
            points = np.concatenate([points, np.zeros((required_points - actual_points, 3))], axis=0)
            points[-1] = np.array([-999, actual_points, -999])

        # visualise
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)

        # o3d.visualization.draw_geometries([pcd])
        # print("breakpoint here")
        return points