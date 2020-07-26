from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

from .sac import SAC
from simple_rl.network import TwinnedStateActionFunction
from simple_rl.utils import soft_update, disable_gradient


class DisCor(SAC):

    def __init__(self, state_shape, action_shape, device, batch_size=256,
                 gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, replay_size=10**6,
                 start_steps=10**4, lr_alpha=3e-4, target_update_coef=5e-3,
                 lr_error=3e-4, tau_init=10.0):
        super().__init__(
            state_shape, action_shape, device, batch_size, gamma, lr_actor,
            lr_critic, replay_size, start_steps, lr_alpha, target_update_coef)

        self.error = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256, 256],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device)
        self.error_target = TwinnedStateActionFunction(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[256, 256, 256],
            HiddenActivation=partial(nn.ReLU, inplace=True)
        ).to(device).eval()

        self.error_target.load_state_dict(self.error.state_dict())
        disable_gradient(self.error_target)

        self.optim_error = torch.optim.Adam(
            self.error.parameters(), lr=lr_error)

        self.tau1 = torch.tensor(tau_init, device=device, requires_grad=False)
        self.tau2 = torch.tensor(tau_init, device=device, requires_grad=False)

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = \
            self.buffer.sample(self.batch_size)

        curr_qs1, curr_qs2, target_qs = self.update_critic(
            states, actions, rewards, dones, next_states)
        self.update_error(
            states, actions, dones, next_states, curr_qs1, curr_qs2, target_qs)
        del curr_qs1
        del curr_qs2
        del target_qs

        self.update_actor(states)
        self.update_target()

    def sample_errors(self, states):
        with torch.no_grad():
            actions, _ = self.actor.sample(states)
            errors1, errors2 = self.error_target(states, actions)
        return errors1, errors2

    def calculate_importance_weights(self, next_states, dones):
        next_errors1, next_errors2 = self.sample_errors(next_states)

        x1 = -(1.0 - dones) * self.gamma * next_errors1 / (self.tau1 + 1e-8)
        x2 = -(1.0 - dones) * self.gamma * next_errors2 / (self.tau2 + 1e-8)

        return F.softmax(x1, dim=0), F.softmax(x2, dim=0)

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards + (1.0 - dones) * self.gamma * next_qs

        imp_ws1, imp_ws2 = \
            self.calculate_importance_weights(next_states, dones)

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mul_(imp_ws1).sum()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mul_(imp_ws1).sum()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

        return curr_qs1.detach_(), curr_qs2.detach_(), target_qs

    def update_error(self, states, actions, dones, next_states, curr_qs1,
                     curr_qs2, target_qs):
        curr_errors1, curr_errors2 = self.error(states, actions)
        next_errors1, next_errors2 = self.sample_errors(next_states)

        with torch.no_grad():
            target_errors1 = (curr_qs1 - target_qs).abs_() \
                + (1.0 - dones) * self.gamma * next_errors1
            target_errors2 = (curr_qs2 - target_qs).abs_() \
                + (1.0 - dones) * self.gamma * next_errors2

        loss_error1 = (curr_errors1 - target_errors1).pow_(2).mean()
        loss_error2 = (curr_errors2 - target_errors2).pow_(2).mean()

        self.optim_error.zero_grad()
        (loss_error1 + loss_error2).backward(retain_graph=False)
        self.optim_error.step()

        mean_errors1 = curr_errors1.detach_().mean()
        mean_errors2 = curr_errors2.detach_().mean()

        self.tau1.data.mul_(1.0 - self.target_update_coef)
        self.tau1.data.add_(self.target_update_coef * mean_errors1.data)
        self.tau2.data.mul_(1.0 - self.target_update_coef)
        self.tau2.data.add_(self.target_update_coef * mean_errors2.data)

    def update_target(self):
        soft_update(
            self.critic_target, self.critic, self.target_update_coef)
        soft_update(
            self.error_target, self.error, self.target_update_coef)
