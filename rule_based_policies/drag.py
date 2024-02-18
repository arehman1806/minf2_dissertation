import numpy as np

from .rule_based_policy import RuleBasedPolicy

class DragPolicy(RuleBasedPolicy):
    def __init__(self, env):
        super().__init__(env)

        self.base_speed = self.policy_params["base_speed"]
        self.kp_angular = self.policy_params["kp_angular"]

    def predict(self, observation, deterministic=True):
        """
        move to the target in reverse direction
        """
        observation = observation["vect_obs"][0]
        
        # yaw angle
        robot_heading = observation[self.proprioception_indices["robot_orientation"] + 2]
        # target distance and angle
        target_distance, target_angle = observation[self.proprioception_indices["polar_robot_target"]: self.proprioception_indices["polar_robot_target"] + 2]

        desired_heading = (target_angle + np.pi) % (2 * np.pi) # Adjust target angle for reverse direction
        heading_error = desired_heading - robot_heading


        # Normalize the heading error to be within -pi to pi
        if heading_error > np.pi:
            heading_error -= 2 * np.pi
        elif heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Angular velocity (ω) control
        angular_velocity = self.kp_angular * heading_error # Tune kp_angular for responsiveness

        # Linear velocity (v), negative for reverse movement. You might want to adjust the speed based on distance
        linear_velocity = -self.base_speed # Keep it simple, or make it dynamic based on target_distance

        # Clip the action to the allowed range of -1 and 1
        velocities = np.clip([linear_velocity, angular_velocity], -1, 1)

        action = np.zeros((self._env.num_envs, 3))
        action[:, 0:2] = velocities

        return action, None