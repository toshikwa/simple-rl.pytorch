import torch
from torch import nn
from torch.optim import Adam

from .base import OffPolicy
from simple_rl.network import DeterministicPolicy, QFunc
from simple_rl.utils import soft_update, disable_gradient


class DDPG(OffPolicy):

    def __init__(self, state_shape, action_shape, device, seed, batch_size=128,
                 gamma=0.99, nstep=1, replay_size=10**6, start_steps=10**4,
                 lr_actor=1e-3, lr_critic=1e-3, std=0.1,
                 target_update_coef=5e-3):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, gamma, nstep,
            replay_size, start_steps)

        self.std = std
        self.target_update_coef = target_update_coef

        self.build_network()

        soft_update(self.actor_target, self.actor, 1.0)
        disable_gradient(self.actor_target)
        soft_update(self.critic_target, self.critic, 1.0)
        disable_gradient(self.critic_target)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

    def build_network(self):
        self.actor = DeterministicPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.actor_target = DeterministicPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic = QFunc(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device)
        self.critic_target = QFunc(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[400, 300],
            hidden_activation=nn.ReLU(inplace=True)
        ).to(self.device).eval()

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
            # Add noises to explore when collecting samples.
            action.add_(torch.randn_like(action) * self.std).clamp_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs = self.critic(states, actions)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_qs = self.critic_target(next_states, next_actions)
        target_qs = rewards + (1.0 - dones) * self.discount * next_qs

        loss_critic = (curr_qs - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        loss_actor = -self.critic(states, self.actor(states)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_target(self):
        soft_update(
            self.actor_target, self.actor, self.target_update_coef)
        soft_update(
            self.critic_target, self.critic, self.target_update_coef)
