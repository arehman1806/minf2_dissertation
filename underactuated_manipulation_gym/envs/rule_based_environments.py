# Packages
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
# Local dependencies
from underactuated_manipulation_gym.envs.base_option_environment import BaseOptionEnvironment


class RuleBasedEnvironment(BaseOptionEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.previous_vels = np.array([0, 0])
        self.previous_joint_commands = np.array(len(self._robot_config["actuators"]["joints"]) * [0])
        self.consecutive_graps = 0
        self.robot_state = None
        self._gripper_enabled = self._robot_config["actuators"]["gripper"]
        self.joint_params = self._robot_config["parameters"]["joint_params"]
        self.policy_params = self._config["policy_parameters"]

    """
    Dummy function to conform to the Gym API
    """
    def _reward(self, observation, proprioception_indices, action):
        raise NotImplementedError
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs})
    

class PickEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
        self.speed = self.policy_params["speed"]
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        joint_positions = observation["vect_obs"][proprioception_indices["joint_position"]: proprioception_indices["joint_position"] + 2]
        targets = np.array([0, 0.5])
        distance = np.linalg.norm(joint_positions - targets)
        if self._is_a_failure_state(observation, proprioception_indices, action):
            done = True
            return -1, done
        if distance < 0.1:
            done = True
            return 1, done
        return 0, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        proprioception_indices = self.robot_state["proprioception_indices"]
        object_state = self.current_object.get_state()
        # convert robot_state to environment_state (i.e. observation)
        # queenie_pos, queenie_orn = self.robot_state["base_pose"]

        # convert to polar coordinates:
        # polar_r, polar_theta = self.cartesian_to_polar_2d(object_state[0][0], object_state[0][1], queenie_pos[0], queenie_pos[1])

        # for now its good enough but maybe extend the vector space to include polar coordinates for closest point on the object

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, proprioception_indices
    
    def _is_a_failure_state(self, observation, proprioception_indices, action):
        angle_bw_contact_norms = observation["vect_obs"][proprioception_indices["normal_angle"]]
        return not abs(angle_bw_contact_norms) > 3 / 4
    

class DragEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        r, theta = observation["vect_obs"][proprioception_indices["polar_robot_target"]: proprioception_indices["polar_robot_target"] + 2]
        if r < 0.1 or self._is_a_failure_state(observation, proprioception_indices, action):
            done = True
            return 1, done
        return 0, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        robot_target_polar_r, robot_target_polar_theta = self.cartesian_to_polar_2d(target_position[0], target_position[1], robot_pose[0], robot_pose[1])
        # add polar coordinates to observation indices
        observation_indices["polar_robot_target"] = len(vect_obs)
        # add polar coordinates to vector observation
        vect_obs = np.append(vect_obs, [robot_target_polar_r, robot_target_polar_theta])

        position, orientation = p.getBasePositionAndOrientation(self.robot.get_ids()[1], physicsClientId=self.client)
        orientation = p.getEulerFromQuaternion(orientation)
        observation_indices["robot_position"] = len(vect_obs)
        vect_obs = np.append(vect_obs, position)
        observation_indices["robot_orientation"] = len(vect_obs)
        vect_obs = np.append(vect_obs, orientation)

        observation = {"image_obs": image_obs, "vect_obs": vect_obs}

        return observation, observation_indices
    
    def _is_a_failure_state(self, observation, proprioception_indices, action):
        angle_bw_contact_norms = observation["vect_obs"][proprioception_indices["normal_angle"]]
        return not abs(angle_bw_contact_norms) > 3 / 4
    

class ReorientEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        rewards = 0
        if not np.any(action[0:2]):
            done = True
            rewards = 1
        return rewards, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        robot_object_polar_r, robot_object_polar_theta = self.cartesian_to_polar_2d(object_pose[0], object_pose[1], robot_pose[0], robot_pose[1])
        observation_indices["polar_robot_object"] = len(vect_obs)
        vect_obs = np.append(vect_obs, [robot_object_polar_r, robot_object_polar_theta])

        position, orientation = p.getBasePositionAndOrientation(self.robot.get_ids()[1], physicsClientId=self.client)
        orientation = p.getEulerFromQuaternion(orientation)
        observation_indices["robot_position"] = len(vect_obs)
        vect_obs = np.append(vect_obs, position)
        observation_indices["robot_orientation"] = len(vect_obs)
        vect_obs = np.append(vect_obs, orientation)

        point_cloud = self.robot_state["point_cloud"]

        observation = {"image_obs": image_obs, "vect_obs": vect_obs, "point_cloud": point_cloud}

        return observation, observation_indices
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        point_cloud = spaces.Box(low=-np.inf, high=np.inf, shape=obs_space_dry_run[0]["point_cloud"].shape, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs, "point_cloud": point_cloud})
    


class PushDeltaEnvironment(RuleBasedEnvironment):
    def __init__(self, config_file, controllers=None, as_subpolicy=False):
        self.impact_point = None
        self.interim_point = None
        super().__init__(config_file, controllers=controllers, as_subpolicy=as_subpolicy)
    
    def _reward(self, observation, proprioception_indices, action):
        done = False
        reward = 0
        # r, theta = observation["vect_obs"][proprioception_indices["polar_object_target"]: proprioception_indices["polar_robot_target"] + 2]
        object_distance_to_target = self._calculate_object_target_distance()
        if not np.any(action[0:2]) or object_distance_to_target < 0.1:
            done = True
            reward = 1
        return reward, done
    
    def get_observation(self):
        self.robot_state = self.robot.get_state()
        image_obs = self.robot_state["camera"]
        vect_obs = self.robot_state["proprioception"]
        observation_indices = self.robot_state["proprioception_indices"]
        object_pose = self.current_object.get_base_pose()[0]
        robot_pose = self.robot.get_base_pose()[0]
        target_position = self.target.get_base_position()

        robot_object_polar_r, robot_object_polar_theta = self.cartesian_to_polar_2d(object_pose[0], object_pose[1], robot_pose[0], robot_pose[1])
        observation_indices["polar_robot_object"] = len(vect_obs)
        vect_obs = np.append(vect_obs, [robot_object_polar_r, robot_object_polar_theta])

        position, orientation = p.getBasePositionAndOrientation(self.robot.get_ids()[1], physicsClientId=self.client)
        orientation = p.getEulerFromQuaternion(orientation)
        observation_indices["robot_position"] = len(vect_obs)
        vect_obs = np.append(vect_obs, position)
        observation_indices["robot_orientation"] = len(vect_obs)
        vect_obs = np.append(vect_obs, orientation)

        # calculate impact point
        # if self.impact_point is None:
        self.impact_point = self.calculate_impact_point(vect_obs, observation_indices)
        observation_indices["impact_point_position"] = len(vect_obs)
        vect_obs = np.append(vect_obs, self.impact_point[0])
        observation_indices["impact_point_orientation"] = len(vect_obs)
        vect_obs = np.append(vect_obs, self.impact_point[1])
        observation_indices["polar_robot_impact_point"] = len(vect_obs)
        vect_obs = np.append(vect_obs, np.array([self.impact_point[2], self.impact_point[3]]))
        observation_indices["polar_robot_drive_until_point"] = len(vect_obs)
        vect_obs = np.append(vect_obs, np.array([self.impact_point[4], self.impact_point[5]]))

        observation_indices["step_i"] = len(vect_obs)
        vect_obs = np.append(vect_obs, self.step_i)

        point_cloud = self.robot_state["point_cloud"]
        if self.interim_point is None:
            self.interim_point = self.calculate_interim_point(vect_obs, observation_indices, point_cloud, self.impact_point[0])
        polar_robot_interim_point_r, polar_robot_interim_point_theta = self.cartesian_to_polar_2d(self.interim_point[0], self.interim_point[1], robot_pose[0], robot_pose[1])
        observation_indices["polat_robot_interim_point"] = len(vect_obs)
        vect_obs = np.append(vect_obs, np.array([polar_robot_interim_point_r, polar_robot_interim_point_theta]))

        observation = {"image_obs": image_obs, "vect_obs": vect_obs, "point_cloud": point_cloud}

        return observation, observation_indices
    
    def calculate_interim_point(self, observation, observation_indices, point_cloud, impact_point_position):
        # if any point in the point cloud falls on the line between the robot and the impact point, then we find the interim point
        # otherwise the interim point is just the halfway point between the robot and the impact point
        robot_pos, robot_orn = self.robot.get_base_pose()
        object_pos, object_orn = self.current_object.get_base_pose()

        useful_points = int(point_cloud[-1][1])
        point_cloud = point_cloud[0:useful_points]
        if len(point_cloud) == 0:
            return impact_point_position

        box_min, box_max = self.compute_bounding_box(point_cloud)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(point_cloud)

        print(f"box_min: {box_min}, box_max: {box_max}")
        if self.line_intersects_box(robot_pos, impact_point_position, box_min, box_max):
            # find the point in the point cloud that intersects the line between the robot and the impact point
            print("line intersects box")
            #Vector from current to target
            direction = impact_point_position - robot_pos
            direction /= np.linalg.norm(direction)  # Normalize
            
            # Find which axis has the largest absolute component in direction
            major_axis = np.argmax(np.abs(direction))
            
            # Determine side of the bounding box to use based on direction
            if direction[major_axis] > 0:
                interim_point = box_max.copy()
            else:
                interim_point = box_min.copy()
            
            safety_margin = self._config["policy_parameters"]["interim_safety_margin"]
            # Add/subtract safety margin
            if major_axis == 0:
                interim_point[1] += safety_margin if direction[1] > 0 else -safety_margin
            elif major_axis == 1:
                interim_point[0] += safety_margin if direction[0] > 0 else -safety_margin
            # interim_point[1] += safety_margin if direction[1] > 0 else -safety_margin
            
            # Adjust interim_point to ensure it's not exactly on the bounding box for other dimensions
            # for i in range(len(interim_point)):
            #     if i != major_axis:
            #         interim_point[i] = robot_pos[i] + direction[i] * np.linalg.norm(impact_point_position - robot_pos) / 2
            print(f"interim_point: {interim_point}")
            return interim_point
        else:
            return impact_point_position

    
    def compute_bounding_box(self, point_cloud):
        """
        Compute the axis-aligned bounding box of a point cloud.
        
        Parameters:
        - point_cloud: An Nx3 numpy array of points.
        
        Returns:
        - A tuple containing the minimum and maximum corners of the bounding box.
        """
        min_corner = np.min(point_cloud, axis=0)
        max_corner = np.max(point_cloud, axis=0)
        return min_corner, max_corner

    def line_intersects_box(self, line_start, line_end, box_min, box_max):
        """
        Check if a line segment intersects an axis-aligned bounding box.
        
        Parameters:
        - line_start, line_end: The endpoints of the line segment.
        - box_min, box_max: The minimum and maximum corners of the bounding box.
        
        Returns:
        - True if the line intersects the box, False otherwise.
        """
        # Cohen-Sutherland-like algorithm adapted for 3D line-box intersection
        for i in range(3):  # Check for each dimension
            if line_start[i] < box_min[i] and line_end[i] < box_min[i]:
                return False
            if line_start[i] > box_max[i] and line_end[i] > box_max[i]:
                return False
        # More detailed intersection tests can go here
        # For simplicity, we return True, indicating potential intersection
        return True

    
    def calculate_impact_point(self, observation, observation_indices):
        robot_pos, robot_orn = self.robot.get_base_pose()
        object_pos, object_orn = self.current_object.get_base_pose()
        target_pos = self.target.get_base_position()

        vector_target_to_object = np.array(object_pos) - np.array(target_pos)
        normalised_target_to_object = vector_target_to_object / np.linalg.norm(vector_target_to_object)
        impact_point = np.array(object_pos) + normalised_target_to_object * self._config["policy_parameters"]["impact_point_distance"]
        vector_object_to_target = np.array(target_pos) - np.array(object_pos)
        normalised_vector_object_to_target = vector_object_to_target / np.linalg.norm(vector_object_to_target)
        dx = normalised_vector_object_to_target[0]
        dy = normalised_vector_object_to_target[1]
        angle = np.arctan2(dy, dx)
        impact_point_orientation = angle
        polar_robot_impact_point_r, polar_robot_impact_point_theta = self.cartesian_to_polar_2d(impact_point[0], impact_point[1], robot_pos[0], robot_pos[1])

        drive_until_point = np.array(object_pos) - normalised_target_to_object * self._config["policy_parameters"]["drive_until_distance"]
        polar_robot_drive_until_point_r, polar_robot_drive_until_point_theta = self.cartesian_to_polar_2d(drive_until_point[0], drive_until_point[1], robot_pos[0], robot_pos[1])

        # print(f"dtp: {drive_until_point}")
        # p.addUserDebugText("impact_point", impact_point, [1, 0, 0], textSize=1, physicsClientId=self.client)
        return impact_point, impact_point_orientation, polar_robot_impact_point_r, polar_robot_impact_point_theta, polar_robot_drive_until_point_r, polar_robot_drive_until_point_theta

    
    def reset_env_memory(self):
        self.impact_point = None
        self.interim_point = None
    
    def _get_observation_space(self):
        obs_space_dry_run = self.get_observation()
        vect_obs_size = obs_space_dry_run[0]["vect_obs"].shape
        image_obs_size = obs_space_dry_run[0]["image_obs"].shape
        min_obs = np.full(vect_obs_size[0], -np.inf)
        max_obs = np.full(vect_obs_size[0], np.inf)
        image_obs = spaces.Box(low=0, high=255, shape=image_obs_size, dtype=np.uint8)
        vect_obs = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)
        point_cloud = spaces.Box(low=-np.inf, high=np.inf, shape=obs_space_dry_run[0]["point_cloud"].shape, dtype=np.float32)
        return spaces.Dict({"image_obs": image_obs, "vect_obs": vect_obs, "point_cloud": point_cloud})

