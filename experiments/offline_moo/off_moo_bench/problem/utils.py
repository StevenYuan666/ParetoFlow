import torch


def constraint(x, lbound, ubound):
    return torch.clamp(x, lbound, ubound)
