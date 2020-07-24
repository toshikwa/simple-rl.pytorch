from functools import partial
import torch
from torch import nn

from .utils import (
    initialize_weights_orthogonal,
    build_mlp
)


class StateFunction(torch.jit.ScriptModule):

    def __init__(self, state_shape, hidden_units=[64, 64],
                 Activation=nn.Tanh):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

    @torch.jit.script_method
    def forward(self, states):
        return self.net(states)


class StateActionFunction(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 Activation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

    @torch.jit.script_method
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class TwinnedStateActionFunction(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 Activation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            Activation=Activation
        ).apply(initialize_weights_orthogonal)

    @torch.jit.script_method
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)
