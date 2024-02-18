from numpy import ndarray
import numpy as np
import sys
from rule_based_policies.rule_based_policy import RuleBasedPolicy


"""
This is an FSM-based meta policy that selects a sub-policy based on the current state of the environment.
"""
class FSMMetaPolicy(RuleBasedPolicy):

    def __init__(self, env, policy_kwargs):
        super().__init__(env)
        self.last_executed_policy = -1
        self.last_policy_failed = False

    def predict(self, observation: ndarray) -> ndarray:
        """
        Selects a sub-policy based on the current state of the environment.
        """
        observation = observation["vect_obs"][0]
        action = np.zeros((self._env.num_envs, 1))
        if self.has_grasped(observation):
            if self.last_executed_policy != 2:
                action[:, 0] = 2
            else:
                action[:, 0] = 3
        else:
            action[:, 0] = 0
        self.last_executed_policy = action[:, 0][0]
        return action, None
        

    def has_grasped(self, observation: ndarray) -> bool:
        return np.any(observation[self.proprioception_indices["contact"]: self.proprioception_indices["contact"] + 3])