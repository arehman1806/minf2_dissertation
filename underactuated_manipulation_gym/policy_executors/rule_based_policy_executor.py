from .base_policy_executor import BasePolicyExecutor
import rule_based_policies as rb


class RuleBasedPolicyExecutor(BasePolicyExecutor):
    def __init__(self, env_config_file, controllers, policy_class_str):
        self.policy_class_str = policy_class_str
        super().__init__(env_config_file, controllers)
        self._model = self._load_model()

    
    def _load_model(self):
        model_class = getattr(rb, self.policy_class_str)
        model = model_class(self._env)
        return model