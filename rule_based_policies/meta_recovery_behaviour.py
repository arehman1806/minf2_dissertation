from numpy import ndarray
import numpy as np
from rule_based_policies.rule_based_policy import RuleBasedPolicy
from collections import defaultdict

actions_dict = {
    "grasp": 0,
    "push": 1,
    "pick": 2,
    "reorient": 4,
    "drag": 3,
    "push_delta": 5
}

str_to_action = {value: key for key, value in actions_dict.items()}
str_to_action[-1] = "Start"

class MetaRecoveryBehaviourPolicy(RuleBasedPolicy):
    def __init__(self, env, policy_kwargs):
        super().__init__(env)
        self.last_executed_policy = -1
        self.last_policy_success = False  # Track if the last policy was successful
        self.policy_failure_count = defaultdict(int)  # Track failures for each policy

    def predict(self, observation: ndarray) -> ndarray:
        """
        Selects a sub-policy based on the FSM logic and current state of the environment.
        """
        action = np.zeros((self._env.num_envs, 1))
        observation = observation["vect_obs"][0]
        self.last_executed_policy = observation[self.proprioception_indices["last_policy"]]
        self.last_policy_success = observation[self.proprioception_indices["last_policy_success"]]
        if self.last_executed_policy == -1:
            self.policy_failure_count = defaultdict(int)
        if not self.last_policy_success:
            self.policy_failure_count[self.last_executed_policy] += 1

        # Determine the next action based on FSM logic
        if self.last_executed_policy == -1:
            # At the start, choose grasp i.e 0
            next_policy = actions_dict["grasp"]
        elif self.last_executed_policy == actions_dict["grasp"]:
            if not self.last_policy_success:
                # If policy 0 (grasp) fails and has been chosen less than 3 times, choose it again
                if self.policy_failure_count[actions_dict["grasp"]] < 1:
                    next_policy = actions_dict["reorient"]
                else:
                    # If policy 1 has already been chosen 3 times, choose 3
                    next_policy = actions_dict["reorient"]
            else:
                # If policy 1 succeeds, choose based on additional conditions or reset
                next_policy = actions_dict["pick"]
        elif self.last_executed_policy == actions_dict["push_delta"]:
            if not self.last_policy_success:
                # If cant push properly then reorient and then push
                next_policy = actions_dict["reorient"]
            else:
                # If pushed successfully, keep pushing
                next_policy = actions_dict["push_delta"]
        elif self.last_executed_policy == actions_dict["pick"]:
            if not self.last_policy_success:
                # If pick fails, reorient and then pick
                if self.policy_failure_count[actions_dict["pick"]] < 3:
                    next_policy = actions_dict["reorient"]
                else:
                    next_policy = actions_dict["push_delta"]
            else:
                # If pick succeeds, then drag
                next_policy = actions_dict["drag"]
        elif self.last_executed_policy == actions_dict["reorient"]:
            if not self.last_policy_success:
                # If reorient fails, then push
                if self.policy_failure_count[actions_dict["reorient"]] < 3:
                    next_policy = actions_dict["reorient"]
                else:
                    next_policy = actions_dict["push_delta"]
            else:
                # If reorient succeeds, then grasp or push depending on previous failures
                if self.policy_failure_count[actions_dict["grasp"]] < 1:
                    next_policy = actions_dict["grasp"]
                else:
                    next_policy = actions_dict["push_delta"]
        elif self.last_executed_policy == actions_dict["drag"]:
            if not self.last_policy_success:
                # If drag fails, then reorient
                next_policy = actions_dict["reorient"]
            else:
                # If drag succeeds, then push
                next_policy = actions_dict["grasp"]
        else:
            # Default to policy 1 if conditions are met or specify other logic
            next_policy = actions_dict["push_delta"] if self.policy_failure_count[1] < 3 else actions_dict["reorient"]
        
        policy_str = str_to_action[next_policy]
        last_policy_str = str_to_action[self.last_executed_policy]
        print(f"Last Policy: {last_policy_str} -> Next policy: {policy_str}")

        action[:, 0] = next_policy
        
        return action, None

    def determine_next_policy(self, observation: ndarray) -> int:
        # Implement logic to determine the next policy based on additional conditions
        # Placeholder logic, replace with your actual conditions
        if self.has_grasped(observation):
            return 2
        return 3

    def check_policy_success(self, observation: ndarray, policy: int) -> bool:
        # Implement logic to check if the policy was successful
        # Placeholder logic, modify based on your success criteria
        return True if policy == 2 else False

    def has_grasped(self, observation: ndarray) -> bool:
        # Your existing has_grasped method
        angle_bw_contact_norms = observation[self.proprioception_indices["normal_angle"]]
        return abs(angle_bw_contact_norms) > 3 / 4
