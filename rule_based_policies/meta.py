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
        return np.array([self._env.action_space.sample() for i in self._env.envs], dtype=np.uint8)
        
