from functools import partial
import torch
from torch import nn

from .utils import build_mlp


class StateFunction(nn.Module):

    def __init__(self, state_shape, hidden_units=[64, 64],
                 HiddenActivation=nn.Tanh):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def forward(self, states):
        return self.net(states)


class StateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 HiddenActivation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class TwinnedStateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=[256, 256],
                 HiddenActivation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net1(x), self.net2(x)


class TwinnedStateActionFunctionWithEncoder(nn.Module):

    def __init__(self, encoder, action_shape, hidden_units=[1024, 1024],
                 HiddenActivation=partial(nn.ReLU, inplace=True)):
        super().__init__()

        self.encoder = nn.ModuleDict({
            'body': encoder.body,
            'linear': encoder.linear
        })
        self.mlp_critic = TwinnedStateActionFunction(
            state_shape=(encoder.feature_dim, ),
            action_shape=action_shape,
            hidden_units=hidden_units,
            HiddenActivation=HiddenActivation
        )

    def _forward_encoder(self, states, skip_body):
        if not skip_body:
            with torch.no_grad():
                states = self.encoder['body'](states)
        return self.encoder['linear'](states)

    def forward(self, states, actions, skip_body=False):
        return self.mlp_critic(
            self._forward_encoder(states, skip_body), actions
        )
