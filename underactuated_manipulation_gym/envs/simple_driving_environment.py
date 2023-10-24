from gymnasium import spaces
import numpy as np
import cv2

from underactuated_manipulation_gym.envs.base_environment import BaseManipulationEnvironment

class DifferentialDriveEnv(BaseManipulationEnvironment):
    
    def __init__(self):
        super().__init__()

    def _reward(self, observation, action):
        # Define your reward function
        distance = self._calculate_robot_object_distance()
        # print(distance)
        if self.previous_distance is None:
            self.previous_distance = distance
        reward = 10*(self.previous_distance - distance)
        self.previous_distance = distance
        x, y, contact = observation
        done = False
        if contact:
            reward += 1000
            done = True
        # print(reward)
        return reward, done
    
    def _calculate_action(self, action):
        # Extract the action components
        v, w_angular = action

        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2

        action = [v_left, v_right, 0, 0, 0]
        return action

    
    def _get_observation(self):
        robot_state = self.robot.get_state()
        object_state = self.current_object.get_state()
        # convert robot_state to environment_state (i.e. observation)
        queenie_pos, queenie_orn = robot_state["base_pose"]
        polar_r, polar_theta = self.cartesian_to_polar_2d(object_state[0][0], object_state[0][1], queenie_pos[0], queenie_pos[1])
        contact_points = robot_state["contact_points"]
        contact = len(contact_points[0]) > 0
        try:
            camera_img = robot_state["camera_rgb"]
            cv2.imwrite("test.png", camera_img)
        except:
            print("no image")
            print(robot_state)

        # convert to polar coordinates:

        return np.array([polar_r, polar_theta, contact])
    
    def _get_observation_space(self):
        # Define the observation space
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    def _get_action_space(self):
        min_linear_vel = -2
        max_linear_vel = 2
        min_angular_vel = -2
        max_angular_vel = 2
        min_neck_joint = -1
        max_neck_joint = 0.21
        min_neck_joint = -0.5
        max_neck_joint = 0.5
        min_action = [min_linear_vel, min_angular_vel]
        max_action = [max_linear_vel, max_angular_vel]
        return spaces.Box(low=np.array(min_action), high=np.array(max_action), dtype=np.float32)