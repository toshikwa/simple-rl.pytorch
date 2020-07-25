from functools import partial
import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[400, 300],
                 HiddenActivation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def forward(self, states):
        means = self.net(states)
        return torch.tanh(means)

    def sample(self, states, std):
        actions = self.forward(states)
        return actions.add_(torch.randn_like(actions) * std).clamp_(-1.0, 1.0)

    def evaluate_log_pi(self, states, actions):
        NotImplementedError


class StateIndependentVarianceGaussianPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[64, 64],
                 HiddenActivation=nn.Tanh):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        means = self.net(states)
        return torch.tanh(means)

    def sample(self, states):
        means = self.net(states)
        actions, log_pis = reparameterize(means, self.log_stds)
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        means = self.net(states)
        return evaluate_lop_pi(means, self.log_stds, actions)


class StateDependentVarianceGaussianPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 HiddenActivation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def forward(self, states):
        means, _ = self.net(states).chunk(2, dim=-1)
        return torch.tanh(means)

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        actions, log_pis = reparameterize(means, log_stds.clamp_(-20, 2))
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return evaluate_lop_pi(means, log_stds.clamp_(-20, 2), actions)
