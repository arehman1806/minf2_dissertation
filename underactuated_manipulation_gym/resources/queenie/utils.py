import pybullet as p
def get_link_index(robot, link_name):
        for joint in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, joint)
            if info[12].decode('utf-8') == link_name:
                return joint
        return -1  # Link not found

def get_joint_index(robot, joint_name):
    for joint in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, joint)
        if info[1].decode('utf-8') == joint_name:
            return joint
    return -1  # Joint not found