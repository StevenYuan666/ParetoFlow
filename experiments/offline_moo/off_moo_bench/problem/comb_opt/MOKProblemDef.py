import numpy as np
import torch


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 3))
    # problems.shape: (batch, problem, 3)
    return problems
