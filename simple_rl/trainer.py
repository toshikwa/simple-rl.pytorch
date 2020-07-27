import os
from time import time
from datetime import timedelta
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, env, env_test, algo, device, log_dir, action_repeat=1,
                 num_steps=10**6, eval_interval=10**4, num_eval_episodes=10,
                 seed=0):
        # Important parameters.
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.device = device
        self.log_dir = log_dir

        # Set seed.
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        self.env_test.seed(2**31-seed)

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.action_repeat = action_repeat
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment and set up the algorithm.
        state = self.env.reset()
        self.algo.reset(state)

        # Iterate until environment step reaches 'num_steps'.
        for steps in range(1, self.num_steps // self.action_repeat + 1):

            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, steps)

            # Update the algorithm whenever ready.
            if self.algo.is_update(steps):
                self.algo.update()

            # Evaluate regularly.
            if steps % self.eval_interval == 0:
                self.evaluate(steps)

    def evaluate(self, steps):
        returns = []
        for _ in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward

            returns.append(episode_return)

        mean_return = np.mean(returns)
        self.writer.add_scalar(
            'return/test', mean_return, steps * self.action_repeat)
        print(f'Num steps: {steps * self.action_repeat:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        """ Time elapsed since the start of training. """
        return str(timedelta(seconds=int(time() - self.start_time)))
