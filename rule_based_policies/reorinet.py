from .rule_based_policy import RuleBasedPolicy


class ReorientPolicy(RuleBasedPolicy):
    """
    A policy that reorients the robot to face the object.
    """

    def __init__(self, env):
        super().__init__(env)

    def predict(self, observation):
        """
        Reorients the robot to face the object.
        """
        return self._env.action_space.sample()