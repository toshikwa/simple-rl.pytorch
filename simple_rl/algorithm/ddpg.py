import torch

from .base import OffPolicy
from simple_rl.network import (
    DeterministicPolicy, StateActionFunction
)
from simple_rl.utils import soft_update, disable_gradient


class DDPG(OffPolicy):

    def __init__(self, state_shape, action_shape, device, replay_size=10**6,
                 start_steps=0, batch_size=256, lr=3e-4, gamma=0.99,
                 std=0.1, target_update_coef=0.005):
        super().__init__(
            state_shape, action_shape, device, replay_size, start_steps)

        self.actor = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300]
        ).to(device)
        self.actor_target = DeterministicPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300]
        ).to(device)

        self.critic = StateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300]
        ).to(device)
        self.critic_target = StateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[400, 300]
        ).to(device).eval()

        self.actor_target.load_state_dict(self.actor.state_dict())
        disable_gradient(self.actor_target)

        self.critic_target.load_state_dict(self.critic.state_dict())
        disable_gradient(self.critic_target)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.learning_steps = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.std = std
        self.target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
            action.add_(torch.randn_like(action) * self.std)
            action.clamp_(-1.0, 1.0)
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
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        loss_critic = (curr_qs - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        actions = self.actor(states)
        loss_actor = -self.critic(states, actions).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_target(self):
        soft_update(
            self.actor_target, self.actor, self.target_update_coef)
        soft_update(
            self.critic_target, self.critic, self.target_update_coef)
