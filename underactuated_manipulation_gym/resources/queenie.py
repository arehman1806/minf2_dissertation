import pybullet as p
import os
import math
import numpy as np

class Queenie_Robot():

    def __init__(self, client) -> None:
        self.client = client

        f_name = os.path.join(os.path.dirname(__file__), "urdfs/queenie_pb.urdf")
        self.robot = p.loadURDF(f_name, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.client)

        self._setup_joint_motors()
        # self._setup_gear_constraints()
        p.setJointMotorControl2(self.robot, 1, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(self.robot, 2, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(self.robot, 4, p.POSITION_CONTROL, targetPosition=0.5, force=20)
        p.setJointMotorControl2(self.robot, 5, p.POSITION_CONTROL, targetPosition=0.5, force=20)
    
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.robot
    
    """
    Resets the robot to the given position and orientation
    """
    def reset(self, position, orientation):
        p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=0)
        p.setJointMotorControl2(self.robot, 9, p.VELOCITY_CONTROL, targetVelocity=0)
        p.setJointMotorControl2(self.robot, 10, p.VELOCITY_CONTROL, targetVelocity=0)
        p.setJointMotorControl2(self.robot, 11, p.VELOCITY_CONTROL, targetVelocity=0)
        p.resetBasePositionAndOrientation(self.robot, position, orientation)
    
    """
    Applies the action to the robot
    """
    def apply_action(self, action):
        v_left, v_right = action
        # wheels
        p.setJointMotorControl2(self.robot, 8, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 9, p.VELOCITY_CONTROL, targetVelocity=v_right)
        p.setJointMotorControl2(self.robot, 10, p.VELOCITY_CONTROL, targetVelocity=v_left)
        p.setJointMotorControl2(self.robot, 11, p.VELOCITY_CONTROL, targetVelocity=v_right)
    
    """
    Returns the observation of the robot in a dictionary format with
    separate keys for camera, joint states, and contact points
    """
    def get_observation(self):
        # Base position and orientation
        base_pose = p.getBasePositionAndOrientation(self.robot)

        # Get the POV of the "camera" link
        camera_link_index = self._get_link_index("camera")
        if camera_link_index != -1:
            link_state = p.getLinkState(self.robot, camera_link_index)
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
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, 
                                                       aspect=float(960) /720, 
                                                       nearVal=0.01, 
                                                       farVal=100)

            # Capture the image from this viewpoint
            width, height, rgb_img, depth_img, _ = p.getCameraImage(960, 720, 
                                                                    viewMatrix=view_matrix,
                                                                    projectionMatrix=proj_matrix,
                                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)

            # Construct observation dictionary
            observation = {
                "base_pose": base_pose,
                "camera_rgb": rgb_img,
                "camera_depth": depth_img
            }
            return observation

        return {"base_pose": base_pose}  # Return base pose only if camera link not found




    def _setup_joint_motors(self):
        # set each joint to velocity control with 0 target velocity and 0 force
        for joint in range(p.getNumJoints(self.robot)):
            p.setJointMotorControl2(self.robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            print(p.getJointInfo(self.robot, joint))
        
    def _setup_gear_constraints(self):
        front_left_wheel = 8
        rear_left_wheel = 10
        front_right_wheel = 9
        rear_right_wheel = 11
        # Set up gear constraints so that wheels on the same side rotate at the same speed:
        c1 = p.createConstraint(self.robot, front_left_wheel, self.robot, rear_left_wheel,
                                    jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c1, gearRatio=1, maxForce=100)

        c2 = p.createConstraint(self.robot, front_right_wheel, self.robot, rear_right_wheel,
                                    jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                    parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        p.changeConstraint(c2, gearRatio=1, maxForce=100)

        c = p.createConstraint(self.robot, 4, self.robot, 5,
                            jointType=p.JOINT_GEAR,
                            jointAxis=[0, 1, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, maxForce=1000)

        return

    def _get_link_index(self, link_name):
        for joint in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, joint)
            if info[12].decode('utf-8') == link_name:
                return joint
        return -1  # Link not found
    

    def rotate_vector(self, vector, theta):
        """Rotates 2D vector by theta degrees"""
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        return rotation_matrix.dot(vector)






if __name__ == "__main__":
    qr = Queenie_Robot(None)
