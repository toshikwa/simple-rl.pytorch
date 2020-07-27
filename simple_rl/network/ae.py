import torch
from torch import nn

from .utils import initialize_weight


class Floatify(nn.Module):

    def forward(self, states):
        assert states.dtype == torch.uint8
        return states.float().div_(255.0)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):

    def __init__(self, C, H, W):
        super().__init__()
        self.shape = (C, H, W)

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class LinearLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class BaseAutoEncoder(torch.jit.ScriptModule):

    def __init__(self, state_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()
        # Size of feature map.
        self.map_size = 43 - 2 * num_layers
        # Dimension of output of conv layers.
        self.last_conv_dim = num_filters * self.map_size * self.map_size
        # Dimension of encoder's feature.
        self.feature_dim = feature_dim
        # Number of filters.
        self.num_filters = num_filters


class Body(BaseAutoEncoder):

    def __init__(self, state_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__(state_shape, feature_dim, num_layers, num_filters)

        self.net = nn.Sequential(
            Floatify(),
            nn.Conv2d(state_shape[0], num_filters, 3, stride=2),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Conv2d(num_filters, num_filters, 3, stride=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)],
            Flatten()
        )

    @torch.jit.script_method
    def forward(self, states):
        return self.net(states)


class Encoder(BaseAutoEncoder):

    def __init__(self, state_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__(state_shape, feature_dim, num_layers, num_filters)

        # Conv layers shared between actor and critic.
        self.body = Body(
            state_shape, feature_dim, num_layers, num_filters
        ).apply(initialize_weight)

        # Linear layer for critic.
        self.linear = LinearLayer(
            input_dim=self.last_conv_dim, output_dim=feature_dim
        ).apply(initialize_weight)

        self.state_shape = state_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

    @torch.jit.script_method
    def forward(self, states):
        return self.linear(self.body(states))


class Decoder(BaseAutoEncoder):

    def __init__(self, state_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__(state_shape, feature_dim, num_layers, num_filters)

        self.net = nn.Sequential(
            nn.Linear(feature_dim, self.last_conv_dim),
            nn.ReLU(inplace=True),
            Unflatten(num_filters, self.map_size, self.map_size),
            *[nn.Sequential(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)],
            nn.ConvTranspose2d(
                num_filters, state_shape[0], 3, stride=2, output_padding=1)
        ).apply(initialize_weight)

    @torch.jit.script_method
    def forward(self, features):
        return self.net(features)
