import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2

from underactuated_manipulation_gym.resources.queenie import Queenie_Robot
from underactuated_manipulation_gym.resources.plane import Plane
from underactuated_manipulation_gym.resources.man_object import ObjectMan
from underactuated_manipulation_gym.resources.object_loader import ObjectLoader

class DifferentialDriveEnv(gym.Env):
    def __init__(self):
        super(DifferentialDriveEnv, self).__init__()

        self.client = p.connect(p.GUI)
        p.setGravity(0,0,-10)
        p.setRealTimeSimulation(0)
        self.robot = Queenie_Robot(self.client)
        self.plane = Plane(self.client)
        self.object_loader = ObjectLoader(self.client, "objects_cuboid")
        self.current_object = None

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.step_i = 0

        self.previous_distance = None


    def reset(self):
        # Reset the environment to its initial state
        self.step_i = 0
        pos = [0, 0, 0.4]
        orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot.reset(pos, orn)
        self.current_object = self.object_loader.change_object()
        for _ in range(100):
            p.stepSimulation()
        # Return the initial observation
        return self._get_observation()

    def step(self, action):
        # Extract the action components
        v, w_angular = action

        v_left = v - w_angular * 0.6 / 2
        v_right = v + w_angular * 0.6 / 2

        action = [v_left, v_right, 0, 0]

        self.robot.apply_action(action)

        
        for i in range(50):
            p.stepSimulation()

        # Get the new observation after taking the action
        observation = self._get_observation()

        # Define your reward and done criteria
        reward = self._reward(observation, action)
        done = self.step_i >=100
        self.step_i += 1

        return observation, reward, done, False,{}
    
    def _reward(self, observation, action):
        # Define your reward function
        distance = self._calculate_robot_object_distance()
        if self.previous_distance is None:
            self.previous_distance = distance
        reward = self.previous_distance - distance
        self.previous_distance = distance
        x, y, contact = observation
        if contact:
            reward += 10
        return reward
    
    def _calculate_robot_object_distance(self):
        object_id = self.current_object.get_ids()[1]
        robot_id = self.robot.get_ids()[1]
        object_link_state = p.getBasePositionAndOrientation(object_id)[0]
        robot_link_state = p.getLinkState(robot_id, 5)[0]
        distance = ((object_link_state[0] - robot_link_state[0]) ** 2 +
                    (object_link_state[1] - robot_link_state[1] ** 2) )** 0.5
        
        return distance

    def _get_observation(self):
        robot_state = self.robot.get_observation()
        # convert robot_state to environment_state (i.e. observation)
        queenie_pos, queenie_orn = robot_state["base_pose"]
        contact_points = robot_state["contact_points"]
        contact = len(contact_points) > 0
        try:
            camera_img = robot_state["camera_rgb"]
            cv2.imwrite("test.png", camera_img)
        except:
            print("no image")
            print(robot_state)

        x, y = queenie_pos[0:2]
        return np.array([x, y, contact])


    def render(self, mode='human'):
        # If you want to visualize the robot's behavior, you can implement this method
        pass

    def close(self):
        self.object_loader.empty_scene()
        # Disconnect from PyBullet
        p.disconnect()

    def seed(self, seed=None):
        # Set the random seed for reproducibility
        pass

    def _parse_config(self):
        pass

    def _get_observation_space(self):
        # Define the observation space
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    
    """
    Define the action space
    """
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