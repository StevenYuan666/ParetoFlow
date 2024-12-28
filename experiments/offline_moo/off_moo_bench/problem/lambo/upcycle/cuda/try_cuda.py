import torch


def try_cuda(*objects, device=None):
    if device is None and torch.cuda.is_available():
        objects = [obj.cuda() for obj in objects]
    if device is not None:
        objects = [obj.to(device) for obj in objects]

    if len(objects) == 1:
        return objects[0]

    return objects
