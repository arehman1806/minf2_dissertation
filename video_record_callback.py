import os
import gymnasium as gym
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

class CustomCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, save_freq: int, save_path: str, 
                 name_prefix: str = "rl_model", save_replay_buffer: bool = False, 
                 save_vecnormalize: bool = False, n_eval_episodes: int = 1, 
                 deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose)
        # VideoRecorder variables
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        # Checkpoint variables
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        # Create save path if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Video recording
        if self.n_calls % self.render_freq == 0:
            self.record_video()

        # Model checkpoint saving
        if self.n_calls % self.save_freq == 0:
            self.save_checkpoint()

        return True

    def record_video(self):
        screens = []
        def grab_screens(_locals, _globals):
            screen = self.eval_env.render()
            screens.append(screen)

        evaluate_policy(self.model, self.eval_env, callback=grab_screens, 
                        n_eval_episodes=self.n_eval_episodes, 
                        deterministic=self.deterministic)
        self.logger.record("trajectory/video", Video(th.ByteTensor([screens]), fps=40), 
                           exclude=("stdout", "log", "json", "csv"))

    def save_checkpoint(self):
        model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
        self.model.save(model_path)
        if self.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")

        if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
            replay_buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl")
            self.model.save_replay_buffer(replay_buffer_path)
            if self.verbose > 1:
                print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

        if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
            vec_normalize_path = os.path.join(self.save_path, f"{self.name_prefix}_vecnormalize_{self.num_timesteps}_steps.pkl")
            self.model.get_vec_normalize_env().save(vec_normalize_path)
            if self.verbose >= 2:
                print(f"Saving model VecNormalize to {vec_normalize_path}")

# Usage example
# custom_callback = CustomCallback(
#     eval_env=your_eval_env, 
#     render_freq=100, 
#     save_freq=50000, 
#     save_path="./path_to_save/",
#     n_eval_episodes=1, 
#     deterministic=True, 
#     verbose=1
# )
