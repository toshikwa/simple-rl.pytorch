import torch
from torch import nn

from .utils import build_mlp


class VFunc(torch.jit.ScriptModule):

    def __init__(self, state_shape, hidden_units=[64, 64],
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    @torch.jit.script_method
    def forward(self, states):
        return self.net(states)


class QFunc(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    @torch.jit.script_method
    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


class TwinnedQFunc(torch.jit.ScriptModule):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    @torch.jit.script_method
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)


class TwinnedQFuncWithEncoder(nn.Module):

    def __init__(self, encoder, action_shape, hidden_units=[1024, 1024],
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.encoder = nn.ModuleDict({
            'body': encoder.body,
            'linear': encoder.linear
        })
        self.mlp_critic = TwinnedQFunc(
            state_shape=(encoder.feature_dim, ),
            action_shape=action_shape,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def _encoder(self, states):
        return self.encoder['linear'](self.encoder['body'](states))

    def forward(self, states, actions):
        return self.mlp_critic(self._encoder(states), actions)

    def without_body(self, conv_features, actions):
        return self.mlp_critic(self.encoder['linear'](conv_features), actions)


class TwinnedQFuncWithDetachedEncoder(TwinnedQFuncWithEncoder):

    def _encoder(self, states):
        with torch.no_grad():
            states = self.encoder['body'](states)
        return self.encoder['linear'](states)
