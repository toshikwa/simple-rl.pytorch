def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False
