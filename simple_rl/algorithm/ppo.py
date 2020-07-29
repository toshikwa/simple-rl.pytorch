import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .base import OnPolicy
from simple_rl.network import StateIndependentGaussianPolicy, VFunc


class PPO(OnPolicy):

    def __init__(self, state_shape, action_shape, device, seed, batch_size=64,
                 gamma=0.995, rollout_length=2048, lr_actor=3e-4,
                 lr_critic=3e-4, num_updates=10, clip_eps=0.2,
                 lambda_gae=0.97, coef_ent=0.0, max_grad_norm=0.5):
        super().__init__(
            state_shape, action_shape, device, seed, batch_size, gamma,
            rollout_length)

        self.build_network()

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.targets = torch.empty(
            (rollout_length, 1), dtype=torch.float, device=device)
        self.advantages = torch.empty(
            (rollout_length, 1), dtype=torch.float, device=device)

        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambda_gae = lambda_gae
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def build_network(self):
        self.actor = StateIndependentGaussianPolicy(
            state_shape=self.state_shape,
            action_shape=self.action_shape,
            hidden_units=[64, 64],
            hidden_activation=nn.Tanh()
        ).to(self.device)
        self.critic = VFunc(
            state_shape=self.state_shape,
            hidden_units=[64, 64],
            hidden_activation=nn.Tanh()
        ).to(self.device)

    def explore(self, state):
        state = torch.tensor(
            state, dtype=self.dtype, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def update(self):
        self.learning_steps += 1

        self.calculate_gae()
        for _ in range(self.num_updates):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(
                    self.buffer.states[idxes],
                    self.targets[idxes]
                )
                self.update_actor(
                    self.buffer.states[idxes],
                    self.buffer.actions[idxes],
                    self.buffer.log_pis[idxes],
                    self.advantages[idxes]
                )

    def update_critic(self, states, targets):
        values = self.critic(states)
        loss_critic = (values - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() \
            - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

    def calculate_gae(self):
        with torch.no_grad():
            values = self.critic(self.buffer.states)

        adv = 0
        for t in reversed(range(self.rollout_length)):
            error = self.buffer.rewards[t] + self.gamma * \
                values[t + 1] * (1 - self.buffer.dones[t]) - values[t]
            adv = error + \
                self.gamma * self.lambda_gae * (1 - self.buffer.dones[t]) * adv
            self.targets[t].copy_(adv + values[t])

        self.advantages.copy_(self.targets - values[:-1])
        mean, std = self.advantages.mean(), self.advantages.std()
        self.advantages.add_(-mean).div_(std + 1e-8)
