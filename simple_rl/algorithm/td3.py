from functools import partial
import torch
from torch import nn

from .ddpg import DDPG
from simple_rl.network import TwinnedStateActionFunction
from simple_rl.utils import disable_gradient


class TD3(DDPG):

    def __init__(self, state_shape, action_shape, device, batch_size=128,
                 gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, replay_size=10**6,
                 start_steps=10**4, std=0.1, update_interval_policy=2,
                 std_target=0.2, clip_noise=0.5, target_update_coef=5e-3):
        super().__init__(
            state_shape, action_shape, device, batch_size, gamma, lr_actor,
            lr_critic, replay_size, start_steps, std, target_update_coef)

        self.update_interval_policy = update_interval_policy
        self.std_target = std_target
        self.clip_noise = clip_noise

    def _build_critic(self):
        self.critic = TwinnedStateActionFunction(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(self.device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(self.device).eval()

        self.critic_target.load_state_dict(self.critic.state_dict())
        disable_gradient(self.critic_target)

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)

        if self.learning_steps % self.update_interval_policy == 0:
            self.update_actor(states)
            self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # Add noises to smoothen the target policy.
            noises = torch.randn_like(next_actions).mul_(
                self.std_target).clamp_(-self.clip_noise, self.clip_noise)
            next_qs1, next_qs2 = self.critic_target(
                next_states, next_actions.add_(noises).clamp_(-1.0, 1.0)
            )

        # Use min(Q1, Q2) to reduce the overestimation of the target.
        target_qs = rewards + (
            1.0 - dones) * self.gamma * torch.min(next_qs1, next_qs2)

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        loss_actor = -self.critic.net1(
            torch.cat([states, self.actor(states)], dim=-1)
        ).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()
