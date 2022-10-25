import torch


def denormalize(x, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1).to(x.device)
    std = torch.tensor(std).reshape(1, 3, 1, 1).to(x.device)
    x = x * std + mean
    return x
