import gym
from gym import spaces
import numpy as np
import random

# Failure probabilities for each action pair (a_t, a_{t+1})
        # This needs to be filled according to your specification
        

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([5, 4])
        self.state = [4, 0]  # Initial state
        
        self.failure_probs = {
            (0, 0): 1.0,
            (0, 1): 0.1,
            (0, 2): 0.3,
            (0, 3): 1,
            (0, 4): 0.0,
            (1, 0): 1.0,
            (1, 1): 1.0,
            (1, 2): 0.1,
            (1, 3): 1.0,
            (1, 4): 0.9,
            (2, 0): 0.6,
            (2, 1): 1.0,
            (2, 2): 1.0,
            (2, 3): 1.0,
            (2, 4): 0.05,
            (3, 0): 0.5,
            (3, 1): 1.0,
            (3, 2): 1.0,
            (3, 3): 1.0,
            (3, 4): 0.05,
            (4, 0): 0.5,
            (4, 1): 1.0,
            (4, 2): 1.0,
            (4, 3): 0.05,
            (4, 4): 0.01,
        }
        
        self.max_steps = 10
        self.current_step = 0

        # Track successful sequences
        self.success_sequence = []
        # Define success criteria sequences
        self.success_criteria = [[0, 1, 2], [4, 3, 4, 3]]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        reward = -2  # Default reward for taking an action
        last_action, num_failures = self.state
        fail_prob = self.failure_probs.get((last_action, action), 0)
        
        if np.random.rand() < fail_prob:
            num_failures = min(num_failures + 1, 3)
            # Do not reset sequence on failure to allow tracking of the sequence across failures
        else:
            if last_action != 4:  # Action 4 does not yield any reward even if successful
                reward += 1
            self.success_sequence.append(action)  # Add successful action to sequence

            # Check for successful completion of criteria sequences
            for criteria in self.success_criteria:
                if self.success_sequence[-len(criteria):] == criteria:
                    reward += 10  # Reward for meeting success criteria
                    self.success_sequence = []  # Optionally reset sequence after success
                    break  # Only reward once per step

        self.state = [action, num_failures]
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        
        return self.state, reward, done, {}

    def reset(self):
        # Do not reset num_failures on reset to persist it across episodes
        self.state = [0, self.state[1]]  # Only reset the last action part of the state
        self.current_step = 0
        self.success_sequence = []  # Reset the success sequence tracker
        return self.state
    
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError("Only 'console' mode is supported.")
        print(f"State: {self.state}")

    def close(self):
        pass


def main():
    # create the custom environment
    env = CustomEnv()

    # Flatten the state space for the Q-table
    state_size = env.observation_space.nvec.prod()
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.99
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):
        # reset the environment
        state = env.reset()
        # Convert state to a single index
        state_index = np.ravel_multi_index(state, env.observation_space.nvec)

        done = False

        for s in range(max_steps):
            # exploration-exploitation tradeoff
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state_index, :])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)
            # Convert new state to a single index
            new_state_index = np.ravel_multi_index(new_state, env.observation_space.nvec)

            # Q-learning algorithm
            qtable[state_index, action] = qtable[state_index, action] + learning_rate * (
                reward + discount_rate * np.max(qtable[new_state_index, :]) - qtable[state_index, action]
            )

            # Update to our new state
            state_index = new_state_index

            # if done, finish episode
            if done:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    # watch trained agent
    state = env.reset()
    state_index = np.ravel_multi_index(state, env.observation_space.nvec)
    done = False
    rewards = 0

    for s in range(max_steps):
        action = np.argmax(qtable[state_index, :])
        new_state, reward, done, info = env.step(action)
        new_state_index = np.ravel_multi_index(new_state, env.observation_space.nvec)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state_index = new_state_index

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
    
