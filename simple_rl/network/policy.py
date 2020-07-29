import torch
from torch import nn

from .utils import (
    build_mlp, reparameterize, evaluate_lop_pi, initialize_weight
)
from .ae import LinearLayer


class Clamp(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, log_stds):
        return log_stds.clamp_(-20, 2)


class DeterministicPolicy(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[400, 300],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        NotImplementedError

    def evaluate_log_pi(self, states, actions):
        NotImplementedError


class StateIndependentGaussianPolicy(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[64, 64],
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        ).apply(initialize_weight)

        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    @torch.jit.script_method
    def forward(self, states):
        return torch.tanh(self.net(states))

    @torch.jit.script_method
    def sample(self, states):
        actions, log_pis = reparameterize(self.net(states), self.log_stds)
        return actions, log_pis

    @torch.jit.script_method
    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class StateDependentGaussianPolicy(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=hidden_units[-1],
            hidden_units=hidden_units[:-1],
            hidden_activation=hidden_activation,
            output_activation=hidden_activation
        ).apply(initialize_weight)

        self.mean = nn.Linear(
            hidden_units[-1], action_shape[0]
        ).apply(initialize_weight)

        self.log_std = nn.Sequential(
            nn.Linear(hidden_units[-1], action_shape[0]),
            Clamp()
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, states):
        return torch.tanh(self.mean(self.net(states)))

    @torch.jit.script_method
    def sample(self, states):
        x = self.net(states)
        return reparameterize(self.mean(x), self.log_std(x))

    @torch.jit.script_method
    def evaluate_log_pi(self, states, actions):
        x = self.net(states)
        return evaluate_lop_pi(self.mean(x), self.log_std(x), actions)


class GaussianPolicyWithDetachedEncoder(nn.Module):

    def __init__(self, encoder, action_shape, hidden_units=[1024, 1024],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.encoder = nn.ModuleDict({
            'body': encoder.body,
            'linear': LinearLayer(
                input_dim=encoder.last_conv_dim,
                output_dim=encoder.feature_dim
            ).apply(initialize_weight)
        })
        self.mlp_actor = StateDependentGaussianPolicy(
            state_shape=(encoder.feature_dim, ),
            action_shape=action_shape,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def _encoder(self, states):
        with torch.no_grad():
            states = self.encoder['body'](states)
        return self.encoder['linear'](states)

    def forward(self, states):
        return self.mlp_actor(self._encoder(states))

    def sample(self, states):
        return self.mlp_actor.sample(self._encoder(states))

    def sample_without_body(self, conv_features):
        return self.mlp_actor.sample(self.encoder['linear'](conv_features))
