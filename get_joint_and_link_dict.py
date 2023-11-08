terminal_output = [
    (0, b'gripper_extension', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'gripper_pole', (0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1),
    (1, b'neck', 0, 7, 6, 1, 0.1, 0.1, -1.0471975511965976, 0.22, 1000.0, 0.05, b'head', (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 0),
    (2, b'neck_x', 0, 8, 7, 1, 0.1, 0.1, -0.5, 0.5, 4.0, 0.05, b'head_x', (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 1),
    (3, b'head_gripper_pole_vertical', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'palm', (0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), 2),
    (4, b'palm_left_finger', 1, 9, 8, 1, 0.0, 0.1, 0.0, 0.1, 1000.0, 0.5, b'left_finger', (0.0, 1.0, 0.0), (0.1, 0.025, 0.0), (0.0, 0.0, 0.0, 1.0), 3),
    (5, b'palm_right_finger', 1, 10, 9, 1, 0.0, 0.1, 0.0, 0.1, 1000.0, 0.5, b'right_finger', (0.0, -1.0, 0.0), (0.1, -0.025, 0.0), (0.0, 0.0, 0.0, 1.0), 3),
    (6, b'camera_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'camera', (0.0, 0.0, 0.0), (0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0), 3),
    (7, b'camera_optical_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'camera_optical', (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, -0.4999999999999999, 0.5, 0.5000000000000001), 6),
    (8, b'left_wheel_hinge', 0, 11, 10, 1, 0.0, 0.0, 0.0, -1.0, 1000.0, 1000.0, b'left_wheel', (2.220446049250313e-16, 2.220446049250313e-16, 1.0), (0.15, 0.3, -0.05), (0.4999999999999999, -0.5, -0.5, 0.5000000000000001), -1),
    (9, b'right_wheel_hinge', 0, 12, 11, 1, 0.0, 0.0, 0.0, -1.0, 1000.0, 1000.0, b'right_wheel', (2.220446049250313e-16, 2.220446049250313e-16, 1.0), (0.15, -0.3, -0.05), (0.4999999999999999, -0.5, -0.5, 0.5000000000000001), -1),
    (10, b'left_wheel_back_hinge', 0, 13, 12, 1, 0.0, 0.0, 0.0, -1.0, 1000.0, 1000.0, b'left_wheel_back', (2.220446049250313e-16, 2.220446049250313e-16, 1.0), (-0.15, 0.3, -0.05), (0.4999999999999999, -0.5, -0.5, 0.5000000000000001), -1),
    (11, b'right_wheel_back_hinge', 0, 14, 13, 1, 0.0, 0.0, 0.0, -1.0, 1000.0, 1000.0, b'right_wheel_back', (2.220446049250313e-16, 2.220446049250313e-16, 1.0), (-0.15, -0.3, -0.05), (0.4999999999999999, -0.5, -0.5, 0.5000000000000001), -1),
]
# Initialize empty dictionaries for joints, links, and a mapping of link indexes to names
joints = {}
links = {}
link_names = {}

# Parse each tuple and fill the dictionaries
for line in terminal_output:
    joint_index = line[0]
    joint_name = line[1].decode()  # Decode the byte string to a normal string
    parent_link_index = line[3]
    child_link_index = line[4]
    link_name = line[12].decode()  # Decode the byte string to a normal string

    # Map joint index to joint name
    joints[joint_index] = joint_name

    # Map link indexes to link names, ensure to not overwrite existing entries
    if parent_link_index not in link_names:
        link_names[parent_link_index] = link_name

    # If it's not the base link (which has no parent and is indicated with a parent link index of -1)
    if parent_link_index != -1:
        # Add child link index to the parent link's list
        links.setdefault(parent_link_index, []).append(child_link_index)

    # If the link has no children, it may not be added to the dictionary, so we ensure it's there
    if child_link_index not in links and child_link_index != -1:
        links[child_link_index] = []

# Display the dictionaries
print("Joints dictionary:")
print(joints)
print("\nLink names dictionary:")
print(link_names)
print("\nLinks dictionary:")
print(links)