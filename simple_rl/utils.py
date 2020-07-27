import torch


def soft_update(target, source, tau):
    """ Update target network using Polyak-Ruppert Averaging. """
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    """ Disable gradient calculations of the network. """
    for param in network.parameters():
        param.requires_grad = False


def preprocess_states(states, bits=5):
    """ Preprocess images to fit into [-0.5, 0.5]. """
    assert states.dtype == torch.uint8
    states = states.float()
    bins = 2 ** bits
    if bits < 8:
        states = torch.floor(states / 2 ** (8 - bits))
    states = states / bins
    states = states + torch.rand_like(states) / bins
    states = states - 0.5
    return states
