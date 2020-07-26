import math
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights_orthogonal(m, gain=1.41):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh()):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_stds, noises):
    # NOTE: We only use multivariate gaussian distribution with diagonal
    # covariance matrix,  which can be viewed as simultaneous distribution of
    # gaussian distributions, p_i(u). So, log_probs = \sum_i log p_i(u).
    return (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) \
        - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)


def calculate_log_pi(log_stds, noises, us):
    # NOTE: Because we calculate actions = tanh(us), we need to correct
    # log_probs. Because tanh(u)' = 1 - tanh(u)**2, we need to substract
    # \sum_i log(1 - tanh(u)**2) from gaussian_log_probs. For numerical
    # stabilities, we use the deformation as below.
    # log(1 - tanh(u)**2)
    # = 2 * log(2 / (exp(u) + exp(-u)))
    # = 2 * (log(2) - log(exp(u) * (1 + exp(-2*u))))
    # = 2 * (log(2) - u - softplus(-2*u))
    return calculate_gaussian_log_prob(log_stds, noises) - (
        2 * (math.log(2) - us - F.softplus(-2 * us))
    ).sum(dim=-1, keepdim=True)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def evaluate_lop_pi(means, log_stds, actions):
    us = atanh(actions)
    noises = (us - means).div_(log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, us)


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + noises * stds
    actions = torch.tanh(us)
    return actions, calculate_log_pi(log_stds, noises, us)
