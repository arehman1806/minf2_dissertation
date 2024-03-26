import pybullet as p
import numpy as np
import math
import cv2
import open3d as o3d
from .sensor import Sensor
from ..utils import get_link_index

from .hha import getHHA


class Camera_Sensor(Sensor):

    def __init__(self, client, robot, sensor_name, sensor_params, robot_params, object_handler=None):
        super().__init__(robot, sensor_name, sensor_params, robot_params)
        self.client = client
        self.dim_obs_space = self._setup_camera()

        self.object_handler = object_handler

    def _setup_camera(self):
        self.camera_link_index = get_link_index(self.robot, "camera")
        self.depth = self._sensor_params["depth"]
        self.hha = self._sensor_params["hha"]
        self.rgb = self._sensor_params["rgb"]
        self.greyscale = self._sensor_params["greyscale"]
        self.semantic = self._sensor_params["semantic"]
        self.camera_near = self._sensor_params["camera_near"]
        self.camera_far = self._sensor_params["camera_far"]
        self.resolution = self._sensor_params["resolution"]
        self.fov = self._sensor_params["fov"]

        self.previous_bodies = []
        self.do_vis = False
        # if "pcd" in self._sensor_params.keys():
        #     self.vis = o3d.visualization.Visualizer()
        #     self.vis.create_window()
        #     pcd = o3d.geometry.PointCloud()
        #     points = np.random.rand(100, 3)  # Dummy point cloud data
        #     pcd.points = o3d.utility.Vector3dVector(points)
        #     self.vis.add_geometry(pcd)
        #     self.do_vis = True
        
        self.current_rgb_image = None
        self.current_depth_image = None
        self.current_semantic_image = None

        self.intrinsic_matrix = self._calculate_intrinsic_matrix()
        dry_run = self.get_observation()
        return dry_run.shape
    
    def get_observation(self):
        # Get the POV of the "camera" link

        link_state = p.getLinkState(self.robot, self.camera_link_index)
        camera_pos = link_state[0]
        camera_orn = link_state[1]

        # Convert quaternion to Euler for getting the forward direction
        euler_angles = p.getEulerFromQuaternion(camera_orn)
        forward_vec = [
            -1 * math.sin(euler_angles[2]),  # Assuming Z-Yaw
            math.cos(euler_angles[2])
        ]

        # Rotate the forward vector by 90 degrees to the right
        rotated_forward_vec = self.rotate_vector(forward_vec, -np.pi/2)
        
        look_at_offset = [0.1 * v for v in rotated_forward_vec]
        camera_target = [camera_pos[0] + look_at_offset[0], camera_pos[1] + look_at_offset[1], camera_pos[2]]


        # Set the PyBullet visualizer camera to the exact position and orientation of the "camera" link
        self.view_matrix = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=self.fov, 
                                                    aspect=float(self.resolution) /self.resolution, 
                                                    nearVal=self.camera_near, 
                                                    farVal=self.camera_far)
        
        
        images = []
        

        # Capture the image from this viewpoint
        width, height, rgb_img, depth_img, semantic_img = p.getCameraImage(self.resolution, self.resolution, 
                                                                viewMatrix=self.view_matrix,
                                                                projectionMatrix=self.proj_matrix,
                                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.current_rgb_image = rgb_img
        self.current_depth_image = depth_img
        self.current_semantic_image = semantic_img

        if self.rgb or self.greyscale:
            rgb_img = rgb_img[:, :, :3]  # This is a 3D array
            # Convert to greyscale if required
            if self.greyscale:
                rgb_img = np.uint8(np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140]))
                rgb_img = np.expand_dims(rgb_img, axis=0)  # Make greyscale a single-channel 3D image
            images.append(rgb_img)

        # Prepare depth image
        if self.depth or self.hha:
            depth = self.camera_far * self.camera_near / (self.camera_far - (self.camera_far - self.camera_near) * depth_img)
            # # if self.do_vis:
            # point_cloud = self._get_point_cloud(depth_img)
            # # vis = input("visualise point cloud? ")
            # # if vis == "y":
            # #     self.visualise_point(point_cloud[0:100])
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(point_cloud)

            # self.vis.update_geometry(pcd)
            # self.vis.poll_events()
            # self.vis.update_renderer()
            # o3d.visualization.draw_geometries([pcd])
            if self.hha:
                hha = getHHA(self.intrinsic_matrix, depth, depth)
                hha = hha.transpose(2, 0, 1)
                images.append(hha)
                cv2.imshow("hha", hha[0])
                cv2.imshow("hha1", hha[1])
                cv2.imshow("hha2", hha[2])
                cv2.waitKey(1)
            # cv2.imwrite("chanel_1.png", hha[0])
            # cv2.imwrite("chanel_2.png", hha[1])
            # cv2.imwrite("chanel_3.png", hha[2])

            if self.depth:
                depth_image_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                # Convert to an unsigned 8-bit integer array
                depth_image_normalized = np.uint8(depth_image_normalized)
                depth_image_normalized = np.expand_dims(depth_image_normalized, axis=0)# Expand the depth image to 3 channels
                # depth_img = np.stack((depth_image_normalized, depth_image_normalized, depth_image_normalized), axis=-1)
                images.append(depth_image_normalized)
            # depth_img = np.expand_dims(depth_image_normalized, axis=2)  # Make depth a single-channel 3D image
        # debug stuff                                                                  
        # cv2.imwrite("test.png", np.squeeze(depth_img))
        # Optionally, add semantic image layer
        if self.semantic:
            semantic_img = np.expand_dims(semantic_img, axis=0)  # Make semantic a single-channel 3D image
        if len(images) == 0:
            return np.array([])
        stacked_observation = np.vstack(images)
        return stacked_observation

        # Change to channel-first format
        observation = np.transpose(observation, (2, 0, 1))

        return observation
        
    def rotate_vector(self, vector, theta):
        """Rotates 2D vector by theta degrees"""
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return rotation_matrix.dot(vector)
    
    def _calculate_intrinsic_matrix(self):

        # Convert FOV from degrees to radians for the calculation
        fov_rad = math.radians(self.fov)
        f_x = self.resolution / (2 * math.tan(fov_rad / 2))
        # f_x = 1 / (math.tan(fov_rad / 2))
        f_y = f_x  # Assuming square pixels (aspect ratio = 1)
        c_x = self.resolution / 2
        c_y = self.resolution / 2

        intrinsic_matrix = np.array([[f_x, 0, c_x],
                                     [0, f_y, c_y],
                                     [0,  0,   1]])   
        return intrinsic_matrix
        # print("intrinsic_matrix: ", intrinsic_matrix)


    def visualise_point(self, point_cloud):
        for body in self.previous_bodies:
            p.removeBody(body, physicsClientId=self.client)
        self.previous_bodies = []
        for point in point_cloud:
            visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
            self.previous_bodies.append(visual_shape_id)
            p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=point, physicsClientId=self.client)

        
    def _calculate_camera_matrices(self):
        pass
    
    def get_render_images(self):
        return self.current_rgb_image, self.current_depth_image
    
    def get_current_depth_image(self):
        return self.current_depth_image
