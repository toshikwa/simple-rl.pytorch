import math
import torch
from torch import nn


def initialize_weights_orthogonal(m, gain=1.41):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              Activation=nn.ReLU):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(Activation())
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_stds, noises):
    return (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) \
        - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)


def calculate_log_pi(log_stds, noises, actions):
    return calculate_gaussian_log_prob(log_stds, noises) \
        - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(stds)
    actions = torch.tanh(means + noises * stds)
    return actions, calculate_log_pi(log_stds, noises, actions)
