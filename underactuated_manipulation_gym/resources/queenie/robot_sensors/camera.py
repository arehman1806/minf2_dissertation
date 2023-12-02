import pybullet as p
import numpy as np
import math
import cv2
from .sensor import Sensor
from ..utils import get_link_index


class Camera_Sensor(Sensor):

    def __init__(self, client, robot, sensor_name, sensor_params):
        super().__init__(robot, sensor_name, sensor_params)
        self.client = client
        self.dim_obs_space = self._setup_camera()

    def _setup_camera(self):
        self.camera_link_index = get_link_index(self.robot, "camera")
        self.depth = self._sensor_params["depth"]
        self.rgb = self._sensor_params["rgb"]
        self.greyscale = self._sensor_params["greyscale"]
        self.semantic = self._sensor_params["semantic"]
        self.camera_near = self._sensor_params["camera_near"]
        self.camera_far = self._sensor_params["camera_far"]
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
        view_matrix = p.computeViewMatrix(camera_pos, camera_target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, 
                                                    aspect=float(84) /84, 
                                                    nearVal=self.camera_near, 
                                                    farVal=self.camera_far)

        # Capture the image from this viewpoint
        width, height, rgb_img, depth_img, semantic_img = p.getCameraImage(84, 84, 
                                                                viewMatrix=view_matrix,
                                                                projectionMatrix=proj_matrix,
                                                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = rgb_img[:, :, :3]  # This is a 3D array

        # Convert to greyscale if required
        if self.greyscale:
            rgb_img = np.dot(rgb_img[..., :3], [0.2989, 0.5870, 0.1140])
            rgb_img = np.expand_dims(rgb_img, axis=2)  # Make greyscale a single-channel 3D image

        # Prepare depth image
        if self.depth:
            depth = self.camera_far * self.camera_near / (self.camera_far - (self.camera_far - self.camera_near) * depth_img)
            depth_image_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Convert to an unsigned 8-bit integer array
            depth_image_normalized = np.uint8(depth_image_normalized)
            depth_image_normalized = np.squeeze(depth_image_normalized)
            depth_img = np.expand_dims(depth_image_normalized, axis=2)  # Make depth a single-channel 3D image

        # Optionally, add semantic image layer
        if self.semantic:
            semantic_img = np.expand_dims(semantic_img, axis=2)  # Make semantic a single-channel 3D image

        # Concatenate depth and/or semantic images
        if self.depth and self.semantic:
            observation = np.concatenate((rgb_img, depth_img, semantic_img), axis=2)
        elif self.depth:
            observation = np.concatenate((rgb_img, depth_img), axis=2)
        elif self.semantic:
            observation = np.concatenate((rgb_img, semantic_img), axis=2)
        else:
            observation = rgb_img

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
