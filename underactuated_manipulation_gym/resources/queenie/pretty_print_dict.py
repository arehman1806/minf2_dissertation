from .utils import format_dict_string_to_table

# Input string representing a dictionary
input_string = "{'chassis': -1, 'gripper_pole': 0, 'head': 1, 'head_x': 2, 'palm': 3, 'left_finger': 4, 'right_finger': 5, 'camera': 6, 'camera_optical': 7, 'left_wheel': 8, 'right_wheel': 9, 'left_wheel_back': 10, 'right_wheel_back': 11}"# Using the function to format the string
format_dict_string_to_table(input_string)
