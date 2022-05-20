import torch


def kl_loss(var, mu, reduction='mean'):
    if reduction == 'mean':
        loss = 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var) / len(mu)
    elif reduction == 'none':
        loss = 0.5 * torch.sum(torch.exp(var) + mu**2 - 1 - var)
    else:
        raise ValueError('Reduction must be "mean" or "none"')

    return loss
