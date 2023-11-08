from gymnasium import spaces
import numpy as np
import cv2
import pybullet as p

from underactuated_manipulation_gym.envs.base_environment import BaseManipulationEnvironment

class RandomURDFsSOEnvironment(BaseManipulationEnvironment):
    
    def __init__(self):
        super().__init__()
        self.previous_vels = np.array([0, 0])
        self.previous_joint_positions = np.array([0, 0])
        self.consecutive_graps = 0
        self.robot_state = None

    def _reward(self, observation, action):
        reward = 0

        # penalize large movements, both planer movement and joint angles
        linear_vel, angular_vel, neck_y, neck_x, gripper = action
        current_vels = np.array([linear_vel, angular_vel])
        diff_vels = abs(current_vels - self.previous_vels)
        reward += -0.01 * np.sum(diff_vels)
        self.previous_vels = current_vels
        current_joint_positions = np.array([neck_y, neck_x])
        diff_joint_positions = abs(current_joint_positions - self.previous_joint_positions)
        reward += -0.9 * np.sum(diff_joint_positions)
        self.previous_joint_positions = current_joint_positions

        # Penalise movement away from the object and reward movement closer to the object
        distance = self._calculate_robot_object_distance()
        if self.previous_distance is None:
            self.previous_distance = distance
        reward += 0.01 * self.previous_distance - distance
        self.previous_distance = distance


        # reward contact to encourage exploration
        contacts = observation["vect_obs"][-3:]
        num_contacts = sum(contacts)
        if num_contacts > 0:
            reward += 0.01 * num_contacts
        
        # reward correct grasp
        angle_bw_contact_norms = self.robot_state["contact_points"][3]
        if abs(angle_bw_contact_norms) > np.pi / 2:
            reward += 1 * abs(angle_bw_contact_norms)
            self.consecutive_graps += 1
        else:
            self.consecutive_graps = 0
        done = self.consecutive_graps > 3

        return reward, done
    
    def _calculate_action(self, action):
        # Extract the action components
        v, w_angular, neck_y, neck_x, gripper = action

        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2

        action = [v_left, v_right, neck_y, neck_x, gripper]
        return action

    
    def _get_observation(self):
        self.robot_state = self.robot.get_state()
        object_state = self.current_object.get_state()
        # convert robot_state to environment_state (i.e. observation)
        queenie_pos, queenie_orn = self.robot_state["base_pose"]
        # convert to polar coordinates:
        polar_r, polar_theta = self.cartesian_to_polar_2d(object_state[0][0], object_state[0][1], queenie_pos[0], queenie_pos[1])
        # proprioception
        proprioception = self.robot_state["proprioception"]
        # contact points
        contacts = self.robot_state["contact_points"]
        contact_palm = int(len(contacts[0]) > 0)
        contact_left_finger = int(len(contacts[1]) > 0)
        contact_right_finger = int(len(contacts[2]) > 0)

        # combine all observations
        vect_obs = np.concatenate((np.array([polar_r, polar_theta], dtype=np.float32), np.array(proprioception, dtype=np.float32), np.array([contact_palm, contact_left_finger, contact_right_finger], np.float32)))
        rgb_image_obs = np.array(self.robot_state["camera_rgb"], dtype=np.uint8)
        depth_image_obs = np.array(self.robot_state["camera_depth"], dtype=np.float32)
        depth_image_obs = np.expand_dims(depth_image_obs, axis=0)
        image_obs = rgb_image_obs
        # Transpose to make it channel-first
        image_obs = np.transpose(image_obs, (2, 0, 1))
        image_obs = np.concatenate((image_obs, depth_image_obs), axis=0)
        observation = {"image_obs": image_obs, "vect_obs": vect_obs}
        
        # near = 0.01  # Near plane
        # far = 7.0  # Far plane
        # depth = far * near / (far - (far - near) * depth_image_obs)
        # depth_image_normalized = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # # Convert to an unsigned 8-bit integer array
        # depth_image_normalized = np.uint8(depth_image_normalized)
        # depth_image_normalized = np.squeeze(depth_image_normalized)
        # # Check the image
        # print(depth_image_normalized.shape)
        # print(depth_image_normalized.dtype)

        # try:
        #     cv2.imwrite("./test.png", np.array(self.robot_state["camera_rgb"], dtype=np.uint8))
        #     cv2.imwrite("./test_depth.png", np.array(depth_image_normalized))
        # except Exception as e:
        #     print(f"no image: {e}")

        return observation
    
    def _get_observation_space(self):
        # Define the observation space
        min_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        max_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        image_obs = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
        # return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    def _get_action_space(self):
        min_linear_vel = -2
        max_linear_vel = 2
        min_angular_vel = -2
        max_angular_vel = 2
        min_neck_joint = -1
        max_neck_joint = 0.21
        min_neck_joint = -0.5
        max_neck_joint = 0.5
        max_gripper = 1
        min_gripper = -1
        min_action = [min_linear_vel, min_angular_vel, min_neck_joint, min_neck_joint, min_gripper]
        max_action = [max_linear_vel, max_angular_vel, max_neck_joint, max_neck_joint, max_gripper]
        return spaces.Box(low=np.array(min_action), high=np.array(max_action), dtype=np.float32)