from abc import ABC, abstractmethod
import torch

from simple_rl.buffer import (
    RolloutBuffer, StateReplayBuffer, PixelReplayBuffer
)


class Algorithm(ABC):

    def __init__(self, state_shape, action_shape, device, batch_size, gamma):
        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.dtype = torch.uint8 if len(state_shape) == 3 else torch.float
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def is_update(self, steps):
        pass

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def reset(self, state):
        pass

    def exploit(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def step(self, env, state, t, steps):
        pass

    @abstractmethod
    def update(self):
        pass


class OnPolicy(Algorithm):

    def __init__(self, state_shape, action_shape, device, batch_size, gamma,
                 rollout_length=10**6):
        super().__init__(
            state_shape, action_shape, device, batch_size, gamma)

        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )
        self.rollout_length = rollout_length

    def reset(self, state):
        self.buffer.reset(state)

    def is_update(self, steps):
        return steps % self.rollout_length == 0

    def step(self, env, state, t, steps):
        t += 1

        action, log_pi = self.explore(state)
        state, reward, done, _ = env.step(action)

        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        if done:
            t = 0
            state = env.reset()

        self.buffer.append(
            state, action, reward, done_masked, log_pi)

        return state, t


class OffPolicy(Algorithm):

    def __init__(self, state_shape, action_shape, device, batch_size, gamma,
                 nstep, replay_size=10**6, start_steps=10**4):
        super().__init__(
            state_shape, action_shape, device, batch_size, gamma)

        Buf = PixelReplayBuffer if len(state_shape) == 3 else StateReplayBuffer
        self.buffer = Buf(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            gamma=gamma,
            nstep=nstep
        )
        self.start_steps = start_steps
        self.discount = gamma ** nstep

    def reset(self, state):
        pass

    def is_update(self, steps):
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps):
        t += 1

        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            action = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        self.buffer.append(
            state, action, reward, done_masked, next_state,
            episode_done=done
        )

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t
