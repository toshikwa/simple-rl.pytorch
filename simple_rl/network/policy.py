import torch
from torch import nn

from .utils import build_mlp, reparameterize, evaluate_lop_pi
from .ae import LinearLayer


class DeterministicPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[400, 300],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        NotImplementedError

    def evaluate_log_pi(self, states, actions):
        NotImplementedError


class StateIndependentVarianceGaussianPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[64, 64],
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        means = self.net(states)
        return torch.tanh(means)

    def sample(self, states):
        actions, log_pis = reparameterize(self.net(states), self.log_stds)
        return actions, log_pis

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentVarianceGaussianPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
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


class StateDependentVarianceGaussianPolicyWithEncoder(nn.Module):

    def __init__(self, encoder, action_shape, hidden_units=[1024, 1024],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.encoder = nn.ModuleDict({
            'body': encoder.body,
            'linear': LinearLayer(
                input_dim=encoder.last_conv_dim,
                output_dim=encoder.feature_dim
            )
        })
        self.mlp_actor = StateDependentVarianceGaussianPolicy(
            state_shape=(encoder.feature_dim, ),
            action_shape=action_shape,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def _forward_encoder(self, states, skip_body):
        if not skip_body:
            with torch.no_grad():
                states = self.encoder['body'](states)
        return self.encoder['linear'](states)

    def forward(self, states, skip_body=False):
        return self.mlp_actor(self._forward_encoder(states, skip_body))

    def sample(self, states, skip_body=False):
        return self.mlp_actor.sample(self._forward_encoder(states, skip_body))
