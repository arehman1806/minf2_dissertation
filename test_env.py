import underactuated_manipulation_gym
import gymnasium as gym
import time
import numpy as np
import pybullet as p
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import time
# Function for moving average
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

env = gym.make("queenie_gym_envs/PickEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/pick_environment.yaml")
env.reset()
j = 0
timesteps = 1000
interface_neck_y = p.addUserDebugParameter("neck_y", -1.57, 1.57, 1.5)
interface_neck_x = p.addUserDebugParameter("neck_x", -1.57, 1.57, 0)
interface_gripper_position = p.addUserDebugParameter("gripper position", -1, 1, 1)
interface_v = p.addUserDebugParameter("v", -2, 2, 0)
interface_w_angular = p.addUserDebugParameter("w", -2, 2, 0)
object_mass = p.addUserDebugParameter("object_mass", 0.1, 1, 0.1)
normal_forces = []
lateral_forces = []
roll_forces = []
contact_lost = True
events = []
neck_y = 1.5

# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.changeDynamics(1, -1, mass=0.9)

# Enable interactive mode
plt.ion()

# Initialize your plots
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="normal forces")
line2, = ax.plot([], [], label="lateral forces")
line2_smooth, = ax.plot([], [], 'r-', label="smoothed lateral forces")  # For smoothed data
line3, = ax.plot([], [], label="roll forces")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Force")
ax.set_xlim(0, timesteps)
# Adjust these limits based on expected force ranges
ax.set_ylim(-2, 2)


for i in range(100):
    p.stepSimulation()
for i in range(timesteps):
    neck_y = neck_y - 0.01 if neck_y > 0 else neck_y
    p.changeDynamics(1, -1, mass=p.readUserDebugParameter(object_mass))
    v = p.readUserDebugParameter(interface_v)
    w_angular = p.readUserDebugParameter(interface_w_angular)
    # neck_y = p.readUserDebugParameter(interface_neck_y)
    neck_x = p.readUserDebugParameter(interface_neck_x)
    # gripper_pos = p.readUserDebugParameter(interface_gripper_position)
    gripper_pos = -1
    action = np.array([v, w_angular, neck_y, neck_x, gripper_pos])
    obs, reward, done, _, _ = env.step(action)  
    contact_points = p.getContactPoints(bodyA=2, bodyB=1, linkIndexA=5)
    contact_points_left = p.getContactPoints(bodyA=2, bodyB=1, linkIndexA=4)
    if len(contact_points) <= 0 and len(contact_points_left) <= 0:
        contact_lost = True
    else:
        contact_lost = False
    # print(contact_points)
    normal_force = 0
    lateral_force = [0, 0, 0]
    roll_force = [0, 0, 0]
    lateral_force_norm = 0
    roll_force_norm = 0
    for contact in contact_points:
        normal_force += contact[9]
        lateral_force_norm += contact[10]
        roll_force_norm += contact[12]
        # lateral_force += contact[11] * contact[10] + contact[13] * contact[12]
        lateral_force[0] += contact[11][0] * contact[10]
        lateral_force[1] += contact[11][1] * contact[10]
        lateral_force[2] += contact[11][2] * contact[10]
        roll_force[0] += contact[13][0] * contact[12]
        roll_force[1] += contact[13][1] * contact[12]
        roll_force[2] += contact[13][2] * contact[12]
    # for contact in contact_points_left:
    #     normal_force += contact[9]
    #     lateral_force_norm += contact[10]
    #     roll_force_norm += contact[12]
    #     # lateral_force += contact[11] * contact[10] + contact[13] * contact[12]
    #     lateral_force[0] += contact[11][0] * contact[10]
    #     lateral_force[1] += contact[11][1] * contact[10]
    #     lateral_force[2] += contact[11][2] * contact[10]
    #     roll_force[0] += contact[13][0] * contact[12]
    #     roll_force[1] += contact[13][1] * contact[12]
    #     roll_force[2] += contact[13][2] * contact[12]
    normal_forces.append(normal_force)
    lateral_forces.append(lateral_force_norm)
    roll_forces.append(roll_force_norm)
    # Compute the smoothed lateral forces only for the last 'window_size' elements
    if i >= 5:  # start smoothing only after enough data points are available
        smoothed_lateral_forces = moving_average(lateral_forces[-5:], window_size=5)
    else:
        smoothed_lateral_forces = [lateral_force_norm] * i  # repeat the value for initial points

    # Update plots
    line1.set_xdata(np.append(line1.get_xdata(), i))
    line1.set_ydata(np.append(line1.get_ydata(), normal_force))
    line2.set_xdata(np.append(line2.get_xdata(), i))
    line2.set_ydata(np.append(line2.get_ydata(), lateral_force_norm))
    # line2_smooth.set_xdata(np.arange(i+1 - len(smoothed_lateral_forces), i+1))  # set x data for smoothed points
    # line2_smooth.set_ydata(smoothed_lateral_forces)  # set y data for smoothed points
    line3.set_xdata(np.append(line3.get_xdata(), i))
    line3.set_ydata(np.append(line3.get_ydata(), roll_force_norm))
    
    # Adjust axes limits dynamically if necessary (optional)
    ax.relim()
    ax.autoscale_view()
    
    plt.draw()
    plt.pause(0.001)
    # if contact_made:
    #     events.append("contact_made")
    # print(f"normal force: {normal_force} lateral force: {lateral_force_norm} roll force: {roll_force_norm} total points: {len(contact_points)}")
# Keep the final plot open
plt.ioff()
plt.show()
# plt.plot(normal_forces, label="normal forces")
# plt.plot(lateral_forces, label="lateral forces")
# plt.plot(roll_forces, label="roll forces")
# plt.legend()
# plt.show()

