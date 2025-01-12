"""
This module contains the implementation of the Flow Matching algorithm.
Referring to the implementation of
https://github.com/jmtomczak/intro_dgm/blob/main/sbgms/fm_example.ipynb
"""

import numpy as np
import torch
import torch.nn as nn


# Define a simple neural network
class VectorFieldNet(nn.Module):
    """
    A simple neural network to approximate the vector field in the Flow Matching algorithm
    """

    def __init__(self, D: int, M: int = 512):
        """
        Initialize the neural network
        :param D: int: the input dimension
        :param M: int: the number of hidden units
        """

        super(VectorFieldNet, self).__init__()

        self.D = D
        self.M = M
        self.net = nn.Sequential(
            nn.Linear(D, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network
        :param x: torch.Tensor: the input tensor
        :return: torch.Tensor: the output tensor
        """

        return self.net(x)


class FlowMatching(nn.Module):
    """
    The Flow Matching algorithm
    """

    def __init__(
        self,
        vnet: nn.Module,
        sigma: float,
        D: int,
        T: int,
        stochastic_euler: bool = False,
        prob_path: str = "icfm",
    ):
        """
        Initialize the Flow Matching algorithm
        :param vnet: nn.Module: the vector field neural network
        :param sigma: float: the sigma parameter
        :param D: int: the input dimension
        :param T: int: the number of steps
        :param stochastic_euler: bool: whether to use the stochastic Euler method
        :param prob_path: str: the probability path
        """
        super(FlowMatching, self).__init__()

        self.vnet = vnet

        self.time_embedding = nn.Sequential(nn.Linear(1, D))

        # other params
        self.D = D

        self.T = T

        self.sigma = sigma

        self.stochastic_euler = stochastic_euler

        assert prob_path in [
            "icfm",
            "fm",
        ], (
            f"Error: The probability path could be either Independent CFM (icfm) "
            f"or Lipman's Flow Matching (fm) but {prob_path} was provided."
        )
        self.prob_path = prob_path

        self.PI = torch.from_numpy(np.asarray(np.pi))

    def log_p_base(
        self, x: torch.Tensor, reduction: str = "sum", dim: int = 1
    ) -> torch.Tensor:
        """
        Calculate the log probability of the base distribution
        :param x: torch.Tensor: the input tensor
        :param reduction: str: the reduction method
        :param dim: int: the dimension
        :return: torch.Tensor: the log probability tensor
        """
        log_p = -0.5 * torch.log(2.0 * self.PI) - 0.5 * x**2.0
        if reduction == "mean":
            return torch.mean(log_p, dim)
        elif reduction == "sum":
            return torch.sum(log_p, dim)
        else:
            return log_p

    def sample_base(self, x_1: torch.Tensor) -> torch.Tensor:
        """
        Sample from the base distribution
        :param x_1: torch.Tensor: the input tensor
        :return: torch.Tensor: the output tensor
        """
        # Gaussian base distribution
        if self.prob_path == "icfm":
            return torch.randn_like(x_1)
        elif self.prob_path == "fm":
            return torch.randn_like(x_1)
        else:
            return None

    def sample_p_t(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from the conditional distribution
        :param x_0: torch.Tensor: the input tensor at time 0
        :param x_1: torch.Tensor: the input tensor at time 1
        :param t: torch.Tensor: the input tensor for time
        :return: torch.Tensor: the output tensor
        """
        if self.prob_path == "icfm":
            mu_t = (1.0 - t) * x_0 + t * x_1
            sigma_t = self.sigma
        elif self.prob_path == "fm":
            mu_t = t * x_1
            sigma_t = t * self.sigma - t + 1.0

        x = mu_t + sigma_t * torch.randn_like(x_1)

        return x

    def conditional_vector_field(
        self, x: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the conditional vector field
        :param x: torch.Tensor: the input tensor
        :param x_0: torch.Tensor: the input tensor at time 0
        :param x_1: torch.Tensor: the input tensor at time 1
        :param t: torch.Tensor: the input tensor for time
        :return: torch.Tensor: the output tensor
        """
        if self.prob_path == "icfm":
            u_t = x_1 - x_0
        elif self.prob_path == "fm":
            u_t = (x_1 - (1.0 - self.sigma) * x) / (1.0 - (1.0 - self.sigma) * t)

        return u_t

    def forward(self, x_1: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Forward pass of the Flow Matching algorithm
        :param x_1: torch.Tensor: the input tensor at time 1
        :param reduction: str: the reduction method
        :return: torch.Tensor: the loss tensor
        """
        # =====Flow Matching
        # =====
        # z ~ q(z), e.g., q(z) = q(x_0) q(x_1), q(x_0) = base, q(x_1) = empirical
        # t ~ Uniform(0, 1)
        x_0 = self.sample_base(
            x_1
        )  # sample from the base distribution (e.g., Normal(0,I))
        t = torch.rand(size=(x_1.shape[0], 1)).to(x_1.device)

        # =====
        # sample from p(x|z)
        x = self.sample_p_t(x_0, x_1, t)  # sample independent rv

        # =====
        # invert interpolation, i.e., calculate vector field v(x,t)
        t_embd = self.time_embedding(t)
        v = self.vnet(x + t_embd)

        # =====
        # conditional vector field
        u_t = self.conditional_vector_field(x, x_0, x_1, t)

        # =====LOSS: Flow Matching
        FM_loss = torch.pow(v - u_t, 2).mean(-1)

        # Final LOSS
        if reduction == "sum":
            loss = FM_loss.sum()
        else:
            loss = FM_loss.mean()

        return loss

    # This is an unconditional sampling process
    def sample(self, batch_size: int = 64) -> torch.Tensor:
        """
        Sample from the Flow Matching algorithm
        :param batch_size: int: the batch size
        :return: torch.Tensor: the output tensor
        """
        # Euler method
        # sample x_0 first
        x_t = self.sample_base(torch.empty(batch_size, self.D)).to(torch.device("cuda"))

        # then go step-by-step to x_1 (data)
        ts = torch.linspace(0.0, 1.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts[1:]:
            t_embedding = self.time_embedding(torch.Tensor([t]).to(x_t.device))
            x_t = x_t + self.vnet(x_t + t_embedding) * delta_t
            # Stochastic Euler method
            if self.stochastic_euler:
                x_t = x_t + torch.randn_like(x_t) * delta_t

        x_final = x_t
        return x_final

    def log_prob(self, x_1: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Calculate the log probability of the Flow Matching algorithm
        :param x_1: torch.Tensor: the input tensor at time 1
        :param reduction: str: the reduction method
        :return: torch.Tensor: the log probability tensor
        """
        # backward Euler (see Appendix C in Lipman's paper)
        ts = torch.linspace(1.0, 0.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts:
            if t == 1.0:
                x_t = x_1 * 1.0
                f_t = 0.0
            else:
                # Calculate phi_t
                t_embedding = self.time_embedding(torch.Tensor([t]).to(x_1.device))
                x_t = x_t - self.vnet(x_t + t_embedding) * delta_t

                # Calculate f_t
                # approximate the divergence using the Hutchinson trace estimator and the autograd
                self.vnet.eval()  # set the vector field net to evaluation

                x = torch.tensor(
                    x_t.data, device=x_1.device
                )  # copy the original data (it doesn't require grads!)
                x.requires_grad = True

                e = torch.randn_like(x).to(x_1.device)  # epsilon ~ Normal(0, I)

                e_grad = torch.autograd.grad(self.vnet(x).sum(), x, create_graph=True)[
                    0
                ]
                e_grad_e = e_grad * e
                f_t = e_grad_e.view(x.shape[0], -1).sum(dim=1)

                self.vnet.eval()  # set the vector field net to train again

        log_p_1 = self.log_p_base(x_t, reduction="sum") - f_t

        if reduction == "mean":
            return log_p_1.mean()
        elif reduction == "sum":
            return log_p_1.sum()
