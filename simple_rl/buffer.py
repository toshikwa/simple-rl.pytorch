from collections import deque
import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._p = 0
        self.buffer_size = buffer_size

        self.states = torch.empty(
            (buffer_size + 1, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size + 1, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)

    def reset(self, state):
        self.states[-1].copy_(torch.from_numpy(state))
        self.dones[-1] = 0

    def append(self, next_state, action, reward, done, log_pi):
        if self._p == 0:
            self.states[0].copy_(self.states[-1])
            self.dones[0].copy_(self.dones[-1])

        self.states[self._p + 1].copy_(torch.from_numpy(next_state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p + 1] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self._p = (self._p + 1) % self.buffer_size


class NStepBuffer:

    def __init__(self, gamma=0.99, nstep=3):
        self.discounts = [gamma ** i for i in range(nstep)]
        self.nstep = nstep
        self.states = deque(maxlen=self.nstep)
        self.actions = deque(maxlen=self.nstep)
        self.rewards = deque(maxlen=self.nstep)

    def append(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self):
        assert len(self.rewards) > 0

        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self.nstep_reward()
        return state, action, reward

    def nstep_reward(self):
        reward = np.sum([r * d for r, d in zip(self.rewards, self.discounts)])
        self.rewards.popleft()
        return reward

    def is_empty(self):
        return len(self.rewards) == 0

    def is_full(self):
        return len(self.rewards) == self.nstep

    def __len__(self):
        return len(self.rewards)


class _ReplayBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device,
                 gamma, nstep):
        self._p = 0
        self._n = 0
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.nstep = nstep

        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)

        if nstep != 1:
            self.nstep_buffer = NStepBuffer(gamma, nstep)

    def append(self, state, action, reward, done, next_state,
               episode_done=None):

        if self.nstep != 1:
            self.nstep_buffer.append(state, action, reward)

            if self.nstep_buffer.is_full():
                state, action, reward = self.nstep_buffer.get()
                self._append(state, action, reward, done, next_state)

            if done or episode_done:
                while not self.nstep_buffer.is_empty():
                    state, action, reward = self.nstep_buffer.get()
                    self._append(state, action, reward, done, next_state)

        else:
            self._append(state, action, reward, done, next_state)

    def _append(self, state, action, reward, done, next_state):
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)


class StateReplayBuffer(_ReplayBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device,
                 gamma, nstep):
        super().__init__(
            buffer_size, state_shape, action_shape, device, gamma, nstep)

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def _append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        super()._append(None, action, reward, done, None)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class PixelReplayBuffer(_ReplayBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device,
                 gamma, nstep):
        super().__init__(
            buffer_size, state_shape, action_shape, device, gamma, nstep)

        self.states = []
        self.next_states = []

    def _append(self, state, action, reward, done, next_state):
        self.states.append(state)
        self.next_states.append(next_state)
        num_excess = len(self.states) - self.buffer_size
        if num_excess > 0:
            del self.states[:num_excess]
            del self.next_states[:num_excess]

        super()._append(None, action, reward, done, None)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)

        # Correct indices for lists of states and next_states.
        bias = -self._p if self._n == self.buffer_size else 0
        state_idxes = np.mod(idxes+bias, self.buffer_size)

        # Convert LazyFrames into np.array.
        for i, idx in enumerate(state_idxes):
            states[i, ...] = self.states[idx]
            next_states[i, ...] = self.next_states[idx]

        return (
            torch.tensor(states, dtype=torch.uint8, device=self.device),
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            torch.tensor(next_states, dtype=torch.uint8, device=self.device)
        )
