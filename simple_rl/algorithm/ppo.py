import numpy as np
import torch
from torch import nn

from simple_rl.network import SteteIndependentGaussianPolicy, StateFunction
from simple_rl.buffer import Buffer


class PPO:

    def __init__(self, state_shape, action_shape, device, lr=3e-4,
                 batch_size=64, gamma=0.995, rollout_length=2048,
                 num_updates=10, clip_eps=0.2, lambda_gae=0.97,
                 coef_ent=0.0, max_grad_norm=0.5):
        super().__init__()

        self.buffer = Buffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            save_log_pi=True
        )

        self.actor = SteteIndependentGaussianPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=[64, 64],
            Activation=nn.Tanh
        ).to(device)
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=[64, 64],
            Activation=nn.Tanh
        ).to(device)

        self.optim_actor = torch.optim.Adam(
            self.actor.parameters(), lr=lr, eps=1e-7)
        self.optim_critic = torch.optim.Adam(
            self.critic.parameters(), lr=lr, eps=1e-7)

        self.learning_steps = 0
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.num_updates = num_updates
        self.clip_eps = clip_eps
        self.lambda_gae = lambda_gae
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, steps):
        return steps % self.rollout_length == 0

    def step(self, env, state, t):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        if t + 1 == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        if done:
            t = 0
            next_state = env.reset()

        self.buffer.append(
            next_state, action, reward, done_masked, log_pi)

        return next_state, t

    def explore(self, state):
        state = torch.tensor(
            state.copy(), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        state = torch.tensor(
            state.copy(), dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def update(self):
        self.learning_steps += 1

        targets, advantages = self.calculate_gae()

        for _ in range(self.num_updates):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)

            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(
                    self.buffer.states[idxes],
                    targets[idxes]
                )
                self.update_actor(
                    self.buffer.states[idxes],
                    self.buffer.actions[idxes],
                    self.buffer.log_pis[idxes],
                    advantages[idxes]
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

        targets = torch.empty_like(self.buffer.rewards)
        adv = torch.zeros_like(targets[0])

        for t in reversed(range(self.rollout_length)):
            error = self.buffer.rewards[t] + self.gamma \
                * values[t + 1] * (1 - self.buffer.dones[t]) - values[t]
            adv = error + \
                self.gamma * self.lambda_gae * (1 - self.buffer.dones[t]) * adv
            targets[t].copy_(adv + values[t])

        advantages = targets - values[:-1]
        mean, std = advantages.mean(), advantages.std()
        advantages = (advantages - mean) / (std + 1e-8)
        return targets, advantages
