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

    def predict(self, observation: ndarray) -> ndarray:
        """
        Selects a sub-policy based on the current state of the environment.
        """
        action = np.zeros((self._env.num_envs, 1))
        if np.any(observation["vect_obs"][0][self.proprioception_indices["contact"]: self.proprioception_indices["contact"] + 3]):
            action[:, 0] = 2
        else:
            action[:, 0] = 0
        return action, None
        
