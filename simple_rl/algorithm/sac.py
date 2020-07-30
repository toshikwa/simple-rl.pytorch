import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .base import OffPolicy
from simple_rl.network import StateDependentGaussianPolicy, TwinnedQFunc
from simple_rl.utils import soft_update, disable_gradient


class SAC(OffPolicy):

    def __init__(self, state_shape, action_shape, device, seed, batch_size=256,
                 gamma=0.99, nstep=1, replay_size=10**6, start_steps=10**4,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4, alpha_init=1.0,
                 target_update_coef=5e-3):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, gamma, nstep,
            replay_size, start_steps)

        self.build_network()
        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.alpha = alpha_init
        self.log_alpha = torch.tensor(
            np.log(self.alpha), device=device, requires_grad=True)
        self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = -float(action_shape[0])

        self.target_update_coef = target_update_coef

    def build_network(self):
        self.actor = StateDependentGaussianPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic = TwinnedQFunc(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic_target = TwinnedQFunc(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[256, 256],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device).eval()

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
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

    def calculate_td_error(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)
        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - dones) * self.discount * next_qs

        return (curr_qs1 - target_qs).abs_(), (curr_qs2 - target_qs).abs_()

    def update_critic(self, states, actions, rewards, dones, next_states):
        td_errors1, td_errors2 = self.calculate_td_error(
            states, actions, rewards, dones, next_states
        )
        loss_critic1 = td_errors1.pow_(2).mean()
        loss_critic2 = td_errors2.pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = self.alpha * log_pis.mean() - torch.min(qs1, qs2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        loss_alpha = -self.log_alpha * (
            self.target_entropy + log_pis.detach_().mean()
        )

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()

        with torch.no_grad():
            self.alpha = self.log_alpha.exp().item()

    def update_target(self):
        soft_update(
            self.critic_target,
            self.critic,
            self.target_update_coef
        )
