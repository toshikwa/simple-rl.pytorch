from functools import partial
import torch
from torch import nn

from .base import OffPolicy
from simple_rl.network import (
    StateDependentVarianceGaussianPolicy, TwinnedStateActionFunction
)
from simple_rl.utils import soft_update, disable_gradient


class SAC(OffPolicy):

    def __init__(self, state_shape, action_shape, device, replay_size=10**6,
                 start_steps=10**4, batch_size=256, lr_actor=3e-4,
                 lr_critic=3e-4, lr_alpha=3e-4, gamma=0.99,
                 target_update_coef=5e-3):
        super().__init__(
            state_shape, action_shape, device, replay_size, start_steps,
            batch_size)

        self.actor = StateDependentVarianceGaussianPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device)
        self.critic = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device)
        self.critic_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device).eval()

        self.critic_target.load_state_dict(self.critic.state_dict())
        disable_gradient(self.critic_target)

        self.optim_actor = torch.optim.Adam(
            self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic)

        self.alpha = 1.0
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(action_shape[0])

        self.learning_steps = 0
        self.device = device
        self.gamma = gamma
        self.target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        loss_alpha = -self.log_alpha * (
            self.target_entropy * log_pis.size(0) + log_pis.detach().mean()
        )

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        self.alpha = self.log_alpha.detach().exp().item()

    def update_target(self):
        soft_update(
            self.critic_target, self.critic, self.target_update_coef)
