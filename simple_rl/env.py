
from collections import deque
import numpy as np
import gym
import dmc2gym

gym.logger.set_level(40)


def make_dmc(domain_name, task_name, action_repeat, frame_stack=3,
             image_size=84):
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        frame_skip=action_repeat
    )
    env = FrameStack(env, k=frame_stack)
    return env


class FrameStack(gym.Wrapper):

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low, k, axis=0),
            high=np.repeat(env.observation_space.high, k, axis=0),
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return LazyFrames(list(self._frames))


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self.dtype = frames[0].dtype

    def _force(self):
        return np.concatenate(
            np.array(self._frames, dtype=self.dtype), axis=0)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]
