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
import pandas as pd
exp_name = "rotational_slip_constant"
# Function for moving average
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

def quaternion_inverse(quaternion):
    q = quaternion
    return [q[0], -q[1], -q[2], -q[3]]

def capture_save_image(timestep):
    # Camera settings
    width, height = 640, 480
    camera_target_position = [0.7, 0, 0.05]  # You may need to adjust this based on your robot's position
    distance = 3.0
    yaw = 0  # This will give a side view; adjust the value as needed for the best side view
    pitch = -30  # This may also need to be adjusted
    roll = 0
    up_axis_index = 2

    # Compute the view matrix from the camera's yaw, pitch, and roll angles
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=camera_target_position,
                                                      distance=distance,
                                                      yaw=yaw,
                                                      pitch=pitch,
                                                      roll=roll,
                                                      upAxisIndex=up_axis_index)
    
    # Compute the projection matrix using a field of view of 60 degrees
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height,
                                               nearVal=0.1, farVal=100.0)
    
    # Capture the image
    _, _, img, _, _ = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix,
                                       projectionMatrix=proj_matrix)
    
    # Convert the PyBullet image (RGBA) to a format suitable for OpenCV or saving
    image = np.array(img)
    # Remove alpha channel
    image = image[:, :, :3]
    
    # Save image (assuming you have cv2)
    cv2.imwrite(f'./results/pick/{exp_name}_{timestep}.png', image)


def unwrap_angles(angles):
    """
    Unwraps a sequence of angles to prevent jumps from +pi to -pi and vice versa,
    making the angle progression continuous.
    
    Parameters:
        angles (np.array): The sequence of angles (in radians) to unwrap.
        
    Returns:
        np.array: The unwrapped sequence of angles.
    """
    unwrapped_angles = np.unwrap(angles)  # numpy's unwrap function is used for this purpose
    return unwrapped_angles


def quaternion_multiply(quaternion1, quaternion2):
    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return [w, x, y, z]

def get_relative_position_and_orientation(id1, linkid1, id2, linkid2):
    pos1, orn1 = p.getLinkState(id1, linkid1)[:2]
    pos2, orn2 = p.getBasePositionAndOrientation(id2)
    
    # Relative position
    rel_pos = np.array(pos2) - np.array(pos1)
    distance = np.linalg.norm(rel_pos)
    
    # Relative orientation
    orn1_inv = quaternion_inverse(orn1)
    rel_orn = quaternion_multiply(orn1_inv, orn2)
    
    # Convert quaternion to Euler angles or another representation if needed
    rel_orn_euler = p.getEulerFromQuaternion(rel_orn)
    
    return distance, rel_orn, rel_orn_euler

env = gym.make("queenie_gym_envs/PickEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/pick_environment.yaml")
env.reset()
j = 0
timesteps = 300
interface_neck_y = p.addUserDebugParameter("neck_y", -1.57, 1.57, 1.5)
interface_neck_x = p.addUserDebugParameter("neck_x", -1.57, 1.57, 0)
interface_gripper_position = p.addUserDebugParameter("gripper position", -1, 1, 1)
interface_v = p.addUserDebugParameter("v", -2, 2, 0)
interface_w_angular = p.addUserDebugParameter("w", -2, 2, 0)
object_mass = p.addUserDebugParameter("object_mass", 0.1, 1, 0.1)
distances = []
force_measurements = []
normalised_torques = np.array([0.0], dtype=np.float64)
smoothed_rolls = []
smoothed_pitchs = []
smoothed_yaws = []
euler_rolls = np.array([0.0], dtype=np.float64)
euler_pitchs = np.array([0.0], dtype=np.float64)
euler_yaws = np.array([0.0], dtype=np.float64)
# New Lists for angular velocities
angular_velocity_rolls = [0]  # Starting with 0 for initial velocity
angular_velocity_pitchs = [0]
angular_velocity_yaws = [0]
events = []
neck_y = 1.5



capture_timesteps = [50, 100, 150, 200]
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.changeDynamics(1, -1, mass=0.9)

# Enable interactive mode
plt.ion()

# Initialize two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

# Setup the first subplot for the angular velocities
ax1.set_ylabel("Δ relative orientation (rad/unit sim time)")
ax1.set_xlim(0, timesteps)
ax1.set_ylim(-0.05, 0.05)
line1, = ax1.plot([], [], label="Δ roll")
line2, = ax1.plot([], [], label="Δ pitch")
line3, = ax1.plot([], [], label="Δ yaw")
ax1.legend()

# Setup the second subplot for the force
ax2.set_xlabel("Step")
ax2.set_ylabel("Force (N)")
ax2.set_ylim(-10, 10)  # Update this as per your expected force value range
line4, = ax2.plot([], [], label="Force", color="magenta")  # 'm-' for magenta color
ax2.legend()

# ax3.set_xlabel("Step")
# ax3.set_ylabel("Force (N)")
# ax3.set_xlim(0, timesteps)
# # Adjust these limits based on expected force ranges
# ax3.set_ylim(-40, 40)  # Update this as per your expected force value range
# line5, = ax3.plot([], [], label="Joint Reaction Force", color="magenta")  # 'm-' for magenta color
# ax3.legend()

# ax4.set_xlabel("Step")
# ax4.set_ylabel("Torque (N.m)")
# ax4.set_xlim(0, timesteps)
# # Adjust these limits based on expected force ranges
# ax4.set_ylim(-5, 5)  # Update this as per your expected force value range
# line6, = ax4.plot([], [], label="Joint Reaction Torque", color="magenta")  # 'm-' for magenta color

force = 0.5
p.enableJointForceTorqueSensor(2, 3, 1)

for i in range(100):
    p.stepSimulation()
for i in range(20):
    v = p.readUserDebugParameter(interface_v)
    w_angular = p.readUserDebugParameter(interface_w_angular)
    # neck_y = p.readUserDebugParameter(interface_neck_y)
    neck_x = p.readUserDebugParameter(interface_neck_x)
    # gripper_pos = p.readUserDebugParameter(interface_gripper_position)
    gripper_pos = -1
    action = np.array([v, w_angular, neck_y, neck_x, force, gripper_pos])
    obs, reward, done, _, _ = env.step(action)
for i in range(timesteps):
    neck_y = neck_y - 0.01 if neck_y > -1.5 else neck_y
    p.changeDynamics(1, -1, mass=p.readUserDebugParameter(object_mass))
    v = p.readUserDebugParameter(interface_v)
    w_angular = p.readUserDebugParameter(interface_w_angular)
    # neck_y = p.readUserDebugParameter(interface_neck_y)
    neck_x = p.readUserDebugParameter(interface_neck_x)
    # gripper_pos = p.readUserDebugParameter(interface_gripper_position)
    gripper_pos = -1
    action = np.array([v, w_angular, neck_y, neck_x, force, gripper_pos])
    obs, reward, done, _, _ = env.step(action)
    
    if i in capture_timesteps:
        capture_save_image(i)

    # Simulate getting new euler angles
    distance, rel_orn, rel_orn_euler = get_relative_position_and_orientation(2, 3, 1, -1)  # Placeholder for your actual function

    # get for torque sensor reading
    wrench = p.getJointState(2, 3)[2]
    force_jrf = wrench[0:3]
    torque = wrench[3:6]
    normalised_force = np.linalg.norm(force_jrf)
    normalised_torque = np.linalg.norm(torque)
    normalised_torques = np.append(normalised_torques, normalised_torque)
    # Update lists
    distances.append(distance)
    euler_rolls = np.append(euler_rolls, rel_orn_euler[0])
    euler_pitchs = np.append(euler_pitchs, rel_orn_euler[1])
    euler_yaws = np.append(euler_yaws, rel_orn_euler[2])
    # euler_rolls.extend(rel_orn_euler[0])
    # euler_pitchs.extend(rel_orn_euler[1])
    # euler_yaws.extend(rel_orn_euler[2])

    # Unwrap angles after obtaining them
    euler_rolls = unwrap_angles(np.array(euler_rolls))
    euler_pitchs = unwrap_angles(np.array(euler_pitchs))
    euler_yaws = unwrap_angles(np.array(euler_yaws))

    if i > 0:  # Ensure there's at least two elements to calculate the difference
        delta_t = 1
        # Use the last two elements for calculation, no need to rely on 'i'
        angular_velocity_rolls.append((euler_rolls[-1] - euler_rolls[-2]) / delta_t)
        angular_velocity_pitchs.append((euler_pitchs[-1] - euler_pitchs[-2]) / delta_t)
        angular_velocity_yaws.append((euler_yaws[-1] - euler_yaws[-2]) / delta_t)

    smoothed_rolls = [0]
    smoothed_pitchs = [0]
    smoothed_yaws = [0]
    # Moving average filter for angular velocities
    if i >= 5:  # Start smoothing once we have enough data
        window_size = 5
        smoothed_rolls = moving_average(np.array(angular_velocity_rolls), window_size)
        smoothed_pitchs = moving_average(np.array(angular_velocity_pitchs), window_size)
        smoothed_yaws = moving_average(np.array(angular_velocity_yaws), window_size)
        normalised_torques = moving_average(np.array(normalised_torques), window_size)

        # Thresholds for detecting slips
        threshold = 0.001  # Example threshold, adjust based on your data
        # Check the last element for slip
        if abs(smoothed_rolls[-1]) > threshold or abs(smoothed_pitchs[-1]) > threshold or abs(smoothed_yaws[-1]) > threshold:
            vel = abs(smoothed_rolls[-1])
            if abs(smoothed_pitchs[-1]) > vel:
                vel = abs(smoothed_pitchs[-1])
                if abs(smoothed_yaws[-1]) > vel:
                    vel = abs(smoothed_yaws[-1])
            print(f"Slip detected at timestep {i}")
            # force += 100 * vel
            force += 0
    
    force_measurements.append(force)
    # Update plots for the first subplot (angular velocities)
    if i > 290:
        line1.set_xdata(np.append(line1.get_xdata(), i))
        line1.set_ydata(np.append(line1.get_ydata(), 0))  # Assuming 'smoothed_rolls' has been defined and updated as before
        line2.set_xdata(np.append(line2.get_xdata(), i))
        line2.set_ydata(np.append(line2.get_ydata(), 0))
        line3.set_xdata(np.append(line3.get_xdata(), i))
        line3.set_ydata(np.append(line3.get_ydata(), 0))
    else:
        line1.set_xdata(np.append(line1.get_xdata(), i))
        line1.set_ydata(np.append(line1.get_ydata(), smoothed_rolls[-1]))  # Assuming 'smoothed_rolls' has been defined and updated as before
        line2.set_xdata(np.append(line2.get_xdata(), i))
        line2.set_ydata(np.append(line2.get_ydata(), smoothed_pitchs[-1]))
        line3.set_xdata(np.append(line3.get_xdata(), i))
        line3.set_ydata(np.append(line3.get_ydata(), smoothed_yaws[-1]))

    # Update plots for the second subplot (force)
    line4.set_xdata(np.append(line4.get_xdata(), i))
    line4.set_ydata(np.append(line4.get_ydata(), force))

    # # Update plots for the third subplot (joint reaction force)
    # line5.set_xdata(np.append(line5.get_xdata(), i))
    # line5.set_ydata(np.append(line5.get_ydata(), normalised_force))

    # # Update plots for the fourth subplot (joint reaction torque)
    # line6.set_xdata(np.append(line6.get_xdata(), i))
    # line6.set_ydata(np.append(line6.get_ydata(), normalised_torques[-1]))

    # Adjust axes limits dynamically if necessary for both subplots
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    # ax3.relim()
    # ax3.autoscale_view()
    # ax4.relim()
    # ax4.autoscale_view()
    
    # Draw both subplots
    plt.draw()
    plt.pause(0.001)
    
    plt.draw()
    plt.pause(0.001)
# Keep the final plot open
plt.ioff()  # Turn off interactive mode

# add vertical line for lost contact and add label for this event
ax1.axvline(x=290, color='r', linestyle='--')
ax1.text(290, 0.03, 'Lost Contact', rotation=90)
ax2.axvline(x=290, color='r', linestyle='--')
ax2.text(290, 5, 'Lost Contact', rotation=90)


# Adjust the layout to prevent overlapping labels and titles
plt.tight_layout()

data = {
    "Step": list(range(len(distances))),
    "Distance": distances,
    "Delta Roll": smoothed_rolls,
    "Delta Pitch": smoothed_pitchs,
    "Delta Yaw": smoothed_yaws,
    "Force": force_measurements
}
df = pd.DataFrame(data)
df.to_csv(f'./results/pick/{exp_name}_data.csv', index=False)



plt.savefig(f'./results/pick/{exp_name}_plot.png')

# Display the figure
plt.show()