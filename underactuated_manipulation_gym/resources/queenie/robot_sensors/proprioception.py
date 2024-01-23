import pybullet as p
import numpy as np
from collections import defaultdict
from .sensor import Sensor
from ..utils import get_link_index, get_joint_index


class Proprioception_Sensor(Sensor):

    def __init__(self, client, robot, sensor_name, sensor_params):
        super().__init__(robot, sensor_name, sensor_params)
        self.client = client

        self._left_finger_index = get_link_index(self.robot, "left_finger")
        self._right_finger_index = get_link_index(self.robot, "right_finger")
        self._palm_index = get_link_index(self.robot, "palm")

        self.dim_obs_space = self._setup_proprioception()


    def _setup_proprioception(self):
        self.joints = self._sensor_params["joints"]
        self.num_joints = len(self.joints)
        self._report_position = self._sensor_params["joint_position"]
        self._report_velocity = self._sensor_params["joint_velocity"]
        self._report_jrf = self._sensor_params["jrf"]
        self._report_jmt = self._sensor_params["jmt"]
        self._report_lin_ang_vel = self._sensor_params["lin_ang_velocity"]
        self._contact_links = self._sensor_params["contact_links"]
        self._report_contact_force = self._sensor_params["contact_force"]
        self._report_normal_angle = self._sensor_params["normal_angle"]
        self._report_normal_angle_palm = self._sensor_params["normal_angle_palm"]

        dry_run, _ = self.get_observation()
        return dry_run.shape
    
    def _calculate_contact_norm(self, contacts):
        norm = np.array([0.0,0.0,0.0])
        for contact in contacts:
            norm += np.array(contact[7])
        norm = norm / np.linalg.norm(norm)
        return norm

    def get_observation(self):
        indices = defaultdict(lambda: -1)
        observation = []

        if self._report_lin_ang_vel:
            lin_vel, ang_vel = p.getBaseVelocity(self.robot, self.client)
            # print(f"lin_vel: {lin_vel}, ang_vel: {ang_vel}")
            observation.append(np.linalg.norm(lin_vel))
            observation.append(np.linalg.norm(ang_vel))
            indices["lin_ang_velocity"] = 0
        
        joint_positions = []
        joint_velocities = []
        jrfs = []
        jmts = []
        joint_states = p.getJointStates(self.robot, self.joints, physicsClientId=self.client) if len(self.joints) > 0 else []
        for joint_state in joint_states:
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
            jrfs.extend(joint_state[2])
            jmts.append(joint_state[3])
        if self._report_position:
            indices["joint_position"] = len(observation)
            observation.extend(joint_positions)
        if self._report_velocity:
            indices["joint_velocity"] = len(observation)
            observation.extend(joint_velocities)
        if self._report_jrf:
            indices["jrf"] = len(observation)
            observation.extend(jrfs)
        if self._report_jmt:
            indices["jmt"] = len(observation)
            observation.extend(jmts)
        
        contact_points_left_finger = None
        contact_points_right_finger = None
        contact_points_palm = None
        contacts = []
        contact_forces = []
        for link in self._contact_links:
            contact_points = p.getContactPoints(bodyA=self.robot, linkIndexA=link, physicsClientId=self.client)
            contacts.append(int(len(contact_points) > 0))
            if link == self._left_finger_index:
                contact_points_left_finger = contact_points
            elif link == self._right_finger_index:
                contact_points_right_finger = contact_points
            else:
                contact_points_palm = contact_points
            if self._report_contact_force:
                contact_force = 0
                for contact_point in contact_points:
                    contact_force += contact_point[9]
                contact_forces.append(contact_force)
        indices["contact"] = len(observation)
        observation.extend(contacts)
        if self._report_contact_force:
            indices["contact_force"] = len(observation)
            observation.extend(contact_forces)
        
        if self._report_normal_angle:
            angle_bw_norms = 0
            if len(contact_points_left_finger) > 0 and len(contact_points_right_finger) > 0:
                left_norm = self._calculate_contact_norm(contact_points_left_finger)
                right_norm = self._calculate_contact_norm(contact_points_right_finger)
                angle_bw_norms = np.arccos(np.clip(np.dot(left_norm, right_norm), -1.0, 1.0))
            indices["normal_angle"] = len(observation)
            observation.append(angle_bw_norms)
        
        if self._report_normal_angle_palm:
            angle = 0
            if len(contact_points_palm) > 0:
                palm_norm = self._calculate_contact_norm(contact_points_palm)
                
                # transform the palm norm from world frame to palm frame and find out the angle it makes with y axis of palm frame
                _, palm_orientation_quat, _, _, _, _  = p.getLinkState(self.robot, self._palm_index, physicsClientId=self.client)
                palm_orientation_euler = p.getEulerFromQuaternion(palm_orientation_quat)
                palm_y_axis_world = np.array(palm_orientation_euler[3:6])
                palm_y_axis_world_normalized = palm_y_axis_world / np.linalg.norm(palm_y_axis_world)
                print(f"palm_y_axis_world_normalized: {palm_y_axis_world_normalized} palm_norm: {palm_norm}")
                cos_theta = np.dot(palm_norm, palm_y_axis_world_normalized)
                angle = np.arccos(cos_theta)
                print("Angle in radians:", angle)
            indices["normal_angle_palm"] = len(observation)
            observation.append(angle)
        
        observation = np.array(observation)
        # print(indices)
        
        return observation, indices