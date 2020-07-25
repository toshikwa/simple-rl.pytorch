from functools import partial
import torch
from torch import nn

from .ddpg import DDPG
from simple_rl.network import TwinnedStateActionFunction


class TD3(DDPG):

    def __init__(self, state_shape, action_shape, device, replay_size=10**6,
                 start_steps=10**4, batch_size=128, lr_actor=1e-3,
                 lr_critic=1e-3, gamma=0.99, std=0.1, policy_update_interval=2,
                 std_target=0.2, clip_noise=0.5, target_update_coef=5e-3):
        super().__init__(
            state_shape, action_shape, device, replay_size, start_steps,
            batch_size, lr_actor, lr_critic, gamma, std, target_update_coef)

        self.policy_update_interval = policy_update_interval
        self.std_target = std_target
        self.clip_noise = clip_noise

    def _build_critic(self, state_shape, action_shape, device):
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device).eval()

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)

        if self.learning_steps % self.policy_update_interval == 0:
            self.update_actor(states)
            self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noises = torch.randn_like(next_actions).mul_(
                self.std_target).clamp_(-self.clip_noise, self.clip_noise)
            next_qs1, next_qs2 = self.critic_target(
                next_states, next_actions.add_(noises).clamp_(-1.0, 1.0)
            )

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
