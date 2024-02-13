from base_policy_executor import BasePolicyExecutor


class RuleBasedPolicyExecutor(BasePolicyExecutor):
    def __init__(self, action_space, policy):
        self.action_space = action_space
        self.policy = policy

    def execute(self, observation):
        return self.policy(observation)