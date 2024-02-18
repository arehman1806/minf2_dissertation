from .rule_based_policy import RuleBasedPolicy

import numpy as np
import math


class ReorientPolicy(RuleBasedPolicy):
    """
    A policy that reorients the robot to face the object.
    """

    def __init__(self, env):
        super().__init__(env)
        self.desired_distance = self.policy_params["desired_distance"]
        self.kp_distance = self.policy_params["kp_linear"]
        self.kp_angular = self.policy_params["kp_angular"]


    def predict(self, observation):
        """
        Reorients the robot to face the object.
        """
        # pc = observation["point_cloud"][0]
        vect_obs = observation["vect_obs"][0]
        # num_useful_points = int(pc[-1][1])
        # pc = pc[0: num_useful_points]
        target_distance, target_angle = vect_obs[self.proprioception_indices["polar_robot_object"]: self.proprioception_indices["polar_robot_object"] + 2]
        robot_position = vect_obs[self.proprioception_indices["robot_position"]: self.proprioception_indices["robot_position"] + 2]
        robot_heading = vect_obs[self.proprioception_indices["robot_orientation"] + 2]
        
        desired_heading = target_angle % (2 * math.pi)

        # Calculate heading error
        heading_error = desired_heading - robot_heading
        # Normalize the error
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi

        # Angular velocity control
        angular_velocity = self.kp_angular * heading_error  # Tune for responsiveness

        # Calculate distance error (how far off from the desired distance we are)
        distance_error = target_distance - self.desired_distance

        # Dynamic linear velocity control, positive for forward, negative for reverse
        # Adjusts based on the distance error; moves forward if too far, backwards if too close
        linear_velocity = self.kp_distance * distance_error

        action = np.ones((self._env.num_envs, 3))
        action[:, 0] = linear_velocity
        action[:, 1] = angular_velocity
        return action, None
