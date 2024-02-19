from .rule_based_policy import RuleBasedPolicy

import numpy as np
import math


class PushDeltaPolicy(RuleBasedPolicy):
    """
    A policy that pushes the object a certain distance to move towards the goal
    """

    def __init__(self, env):
        super().__init__(env)
        # self.desired_distance = self.policy_params["desired_distance"]
        self.kp_distance = self.policy_params["kp_linear"]
        self.kp_angular = self.policy_params["kp_angular"]
        self.base_speed = self.policy_params["base_speed"]
        self.distance_threshold = self.policy_params["distance_threshold"]
        self.drive_until_distance = self.policy_params["drive_until_distance"]
        self.step_i = 0
        self.stage_1_i = 0
        self.stage = 0
        self.stop_position = False


    def predict(self, observation, deterministic=True):
        """
        Reorients the robot to face the object, adding the possibility of driving in reverse if it's a shorter path.
        """
        action = np.ones((self._env.num_envs, 3))  # Initialize the action array
        vect_obs = observation["vect_obs"][0]  # Extract vector observations
        step_i_current = vect_obs[self.proprioception_indices["step_i"]]
        if self.step_i > step_i_current:
            self.stage = 0
            self.stage_1_i = 0
            self.stop_position = False
        self.step_i = step_i_current


        if self.stage == 0:
            target_distance, target_angle = vect_obs[self.proprioception_indices["polat_robot_interim_point"]: self.proprioception_indices["polat_robot_interim_point"] + 2]
        elif self.stage == 1:
            target_distance, target_angle = vect_obs[self.proprioception_indices["polar_robot_impact_point"]: self.proprioception_indices["polar_robot_impact_point"] + 2]
        elif self.stage == 2:
            if self.stage_1_i >= self.drive_until_distance * 50:
                action[:, 0] = 0
                action[:, 1] = 0
                return action, None
            self.stage_1_i += 1
            target_distance, target_angle = vect_obs[self.proprioception_indices["polar_robot_drive_until_point"]: self.proprioception_indices["polar_robot_drive_until_point"] + 2]

        robot_position = vect_obs[self.proprioception_indices["robot_position"]: self.proprioception_indices["robot_position"] + 2]
        robot_heading = vect_obs[self.proprioception_indices["robot_orientation"] + 2]
        goal_orientation = vect_obs[self.proprioception_indices["impact_point_orientation"]]

        # Adjust target angle to decide the driving direction
        target_angle_rel = ((target_angle - robot_heading + math.pi) % (2 * math.pi)) - math.pi
        driving_reverse = abs(target_angle_rel) > math.pi / 2
        # print(f"driving_reverse: {driving_reverse}")

        if driving_reverse:
            desired_heading = (target_angle + math.pi) % (2 * math.pi)
        else:
            desired_heading = target_angle

        # Position control
        if target_distance > (self.distance_threshold):  # Distance threshold determines when to switch focus to orientation
            heading_error = desired_heading - robot_heading
            # print("True")
        else:
            # Orientation control
            # print("Not True")
            desired_orientation = goal_orientation
            heading_error = desired_orientation - robot_heading
            if self.stage == 0:
                heading_error = 0.01

        # Normalize the error
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
            # print("heading error > pi")
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi
            # print("heading error < -pi")

        # Angular velocity control
        angular_velocity = self.kp_angular * heading_error

        # Linear velocity control, with reverse driving logic
        if target_distance > self.distance_threshold:
            if driving_reverse:
                linear_velocity = -min(self.base_speed, self.kp_distance * target_distance)  # Reverse if shorter
            else:
                linear_velocity = min(self.base_speed, self.kp_distance * target_distance)  # Forward otherwise
        else:
            # Slow down as orientation becomes the priority
            linear_velocity = 0

        # Update stage to 1 when close to the target and orientation is nearly aligned
        if target_distance < self.distance_threshold and abs(heading_error) < self.distance_threshold:
            # print(f"stage {self.stage} -> {self.stage + 1}")
            self.stage += 1
            self.stop_position = False

        # print(f"heading_error: {heading_error}, target_distance: {target_distance}, stage: {self.stage}")
        action[:, 0] = linear_velocity
        action[:, 1] = angular_velocity
        return action, None

        return action, None
        pc = observation["point_cloud"][0]
        num_useful_points = int(pc[-1][1])
        pc = pc[0: num_useful_points]
        vect_obs = observation["vect_obs"][0]
        target_distance, target_angle = vect_obs[self.proprioception_indices["polar_robot_object"]: self.proprioception_indices["polar_robot_object"] + 2]
        robot_position = vect_obs[self.proprioception_indices["robot_position"]: self.proprioception_indices["robot_position"] + 2]
        robot_heading = vect_obs[self.proprioception_indices["robot_orientation"] + 2]

        impact_point_position = vect_obs[self.proprioception_indices["impact_point_position"]: self.proprioception_indices["impact_point_position"] + 2]
        print(f"impact_point_position: {impact_point_position}")
        impact_point_orientation_vector = vect_obs[self.proprioception_indices["impact_point_orientation"]: self.proprioception_indices["impact_point_orientation"] + 2]

        # Calculate the desired orientation based on the orientation vector
        desired_orientation = np.arctan2(impact_point_orientation_vector[1], impact_point_orientation_vector[0])

        # Calculate the orientation error
        orientation_error = desired_orientation - robot_heading
        # Normalize the error
        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi

        # Decide whether to go forward or reverse based on the orientation error
        # If the orientation error is larger than 90 degrees (pi/2 radians), it might be more efficient to reverse
        if abs(orientation_error) > np.pi / 2:
            # Adjust orientation error for reversing
            orientation_error = (orientation_error - np.pi) % (2 * np.pi) - np.pi
            angular_velocity = self.kp_angular * orientation_error
            # Use negative linear velocity to reverse
            linear_velocity = -self.kp_distance * target_distance
        else:
            angular_velocity = self.kp_angular * orientation_error
            linear_velocity = self.kp_distance * target_distance

        # Calculate distance to the impact point
        distance_to_impact_point = np.linalg.norm(impact_point_position - robot_position)

        # Linear velocity control, adjust to not overshoot the impact point
        linear_velocity = np.clip(linear_velocity, -0.5, 0.5)

        # Stop moving if close enough (within a threshold) to the impact point
        if distance_to_impact_point < 0.1:  # Threshold for stopping, adjust as needed
            linear_velocity = 0
            angular_velocity = 0

        action = np.ones((self._env.num_envs, 3))
        action[:, 0] = linear_velocity
        action[:, 1] = angular_velocity
        return action, None
