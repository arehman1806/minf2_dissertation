import numpy as np

from .rule_based_policy import RuleBasedPolicy

class PickPolicy(RuleBasedPolicy):
    def __init__(self, env):
        super().__init__(env)
        self.speed = self._env.envs[0].speed
        self.proprioception_indices = self._env.envs[0].proprioception_indices
        self.joint_param = self._env.envs[0].joint_params
        self.allowed_delta = np.array([(joint_values["max"] - joint_values["min"]) * self.speed for joint_values in self.joint_param.values()])

    def predict(self, observation, deterministic=True):
        """
        Picks up the object over the head
        """
        joint_positions = observation["vect_obs"][0][self.proprioception_indices["joint_position"]: self.proprioception_indices["joint_position"] + 2]
        target = np.array([-1, 0])
        delta_position = ((target - joint_positions) / self.speed) * (target - joint_positions) 
        joint_position_command = joint_positions + delta_position
        # print(f"{joint_positions} -> {joint_position_command}")
        # obs = observation["vect_obs"]
        # print(f"observed joint positions: {obs[0]}")
        

        # for now lets just set it to left it to maximum angle
        action = np.zeros((self._env.num_envs, self._env.action_space.shape[0]))
        # get the robot position
        
        # zero speed
        action[:, 0] = 0.0
        action[:, 1] = 0.0
        # maximum angle
        action[:, 2:4] = target
        # gripper closed
        action[:, 4] = 0.0
        return action, None

