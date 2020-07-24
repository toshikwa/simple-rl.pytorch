from functools import partial
import torch
from torch import nn

from .utils import (
    initialize_weights_orthogonal,
    build_mlp,
    reparameterize,
    evaluate_lop_pi
)


class SteteIndependentGaussianPolicy(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[64, 64],
                 Activation=nn.Tanh):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

        self.net[-1].apply(
            partial(initialize_weights_orthogonal, gain=1.41 * 1e-2)
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    @torch.jit.script_method
    def forward(self, states):
        means = self.net(states)
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, states):
        means = self.net(states)
        actions, log_pis = reparameterize(means, self.log_stds)
        return actions, log_pis

    @torch.jit.script_method
    def evaluate_log_pi(self, states, actions):
        means = self.net(states)
        return evaluate_lop_pi(means, self.log_stds, actions)


class SteteDependentGaussianPolicy(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 Activation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

    @torch.jit.script_method
    def forward(self, states):
        means, _ = torch.chunk(self.net(states), 2, dim=-1)
        return torch.tanh(means)

    @torch.jit.script_method
    def sample(self, states):
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        actions, log_pis = reparameterize(means, log_stds.clamp_(-20, 2))
        return actions, log_pis

    @torch.jit.script_method
    def evaluate_log_pi(self, states, actions):
        means, log_stds = torch.chunk(self.net(states), 2, dim=-1)
        return evaluate_lop_pi(means, log_stds, actions)
