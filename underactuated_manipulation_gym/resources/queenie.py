import pybullet as p
import os

class Queenie_Robot():

    def __init__(self, client) -> None:
        self.client = client

        f_name = os.path.join(os.path.dirname(__file__), "urdfs/queenie_pb.urdf")
        self.robot = p.loadURDF(f_name, [0, 0, 0.1], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.client)

        self._setup_joint_motors()
        # self._setup_gear_constraints()
    
    """
    Returns the client and robot ids
    """
    def get_ids(self):
        return self.client, self.robot
    
    """
    Resets the robot to the given position and orientation
    """
    def reset(self, position, orientation):
        self._setup_joint_motors()
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
        return p.getBasePositionAndOrientation(self.robot)

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




if __name__ == "__main__":
    qr = Queenie_Robot(None)
