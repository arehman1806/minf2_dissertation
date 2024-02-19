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


    def predict(self, observation, deterministic=True):
        vect_obs = observation["vect_obs"][0]
        target_distance, target_angle = vect_obs[self.proprioception_indices["polar_robot_object"]: self.proprioception_indices["polar_robot_object"] + 2]
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
        angular_velocity = self.kp_angular * heading_error

        # Calculate distance error
        distance_error = target_distance - self.desired_distance

        # Dynamic linear velocity control
        linear_velocity = self.kp_distance * distance_error

        # Consider reducing linear velocity as the angular correction increases
        if abs(heading_error) > 0.1:  # Arbitrary threshold; adjust based on your needs
            linear_velocity *= (1 - min(abs(heading_error) / math.pi, 1))  # Scale down linear velocity based on heading error

        action = np.ones((self._env.num_envs, 3))
        action[:, 0] = linear_velocity
        action[:, 1] = angular_velocity

        # Stop condition based on a more forgiving distance_error threshold
        if abs(distance_error) < 0.2 and abs(heading_error) < 0.1:  # Adjusted threshold
            action[:, 0] = 0  # Stop moving
            action[:, 1] = 0

        return action, None

