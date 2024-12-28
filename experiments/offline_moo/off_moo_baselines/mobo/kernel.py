import torch
from gpytorch.kernels import Kernel

from offline_moo.off_moo_baselines.mobo.mobo_utils import tkwargs
from offline_moo.off_moo_baselines.util.data_structure import FeatureCache

feature_cache = FeatureCache()


class OrderKernel(Kernel):
    has_lengthscale = True

    def forward(self, X, X2, **params):
        global feature_cache
        if len(X.shape) > 2:
            assert X.shape[0] == X2.shape[0]
            batch_size = X.shape[0]

            x1 = feature_cache.push(X).to(**tkwargs)
            x2 = feature_cache.push(X2).to(**tkwargs)

            mat = (x1.unsqueeze(2) - x2.unsqueeze(1)).pow(2).sum(dim=-1)
            mat = torch.exp(-self.lengthscale * mat)

            mat = mat.view(batch_size, -1, mat.shape[-1])
            return mat
        else:
            mat = torch.zeros((len(X), len(X2))).to(**tkwargs)
            x1 = []
            for i in range(len(X)):
                x1.append(feature_cache.push(X[i]))
            x2 = []
            for j in range(len(X2)):
                x2.append(feature_cache.push(X2[j]))

            x1 = torch.vstack(x1).to(**tkwargs)
            x2 = torch.vstack(x2).to(**tkwargs)
            x1 = torch.reshape(x1, (x1.shape[0], 1, -1)).to(**tkwargs)
            x2 = torch.reshape(x2, (1, x2.shape[0], -1)).to(**tkwargs)
            x1 = torch.tile(x1, (1, x2.shape[0], 1)).to(**tkwargs)
            x2 = torch.tile(x2, (x1.shape[0], 1, 1)).to(**tkwargs)
            mat = torch.sum((x1 - x2) ** 2, dim=-1).to(**tkwargs)
            mat = torch.exp(-self.lengthscale * mat)
            return mat


class CategoricalOverlap(Kernel):
    """Implementation of the categorical overlap kernel.
    This is the most basic form of the categorical kernel that essentially invokes a Kronecker delta function
    between any two elements.
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super(CategoricalOverlap, self).__init__(has_lengthscale=True, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # First, convert one-hot to ordinal representation

        diff = x1[:, None] - x2[None, :]
        # nonzero location = different cat
        diff[torch.abs(diff) > 1e-5] = 1
        # invert, to now count same cats
        diff1 = torch.logical_not(diff).float()
        if self.ard_num_dims is not None and self.ard_num_dims > 1:
            k_cat = torch.sum(self.lengthscale * diff1, dim=-1) / torch.sum(
                self.lengthscale
            )
        else:
            # dividing by number of cat variables to keep this term in range [0,1]
            k_cat = torch.sum(diff1, dim=-1) / x1.shape[1]
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)


class TransformedCategorical(CategoricalOverlap):
    """
    Second kind of transformed kernel of form:
    $$ k(x, x') = \exp(\frac{\lambda}{n}) \sum_{i=1}^n [x_i = x'_i] )$$ (if non-ARD)
    or
    $$ k(x, x') = \exp(\frac{1}{n} \sum_{i=1}^n \lambda_i [x_i = x'_i]) $$ if ARD
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, exp="rbf", **params):
        if x1.dim() <= 3:
            # expand x1 and x2 to calc hamming distance
            M1_expanded = x1.unsqueeze(2)
            M2_expanded = x2.unsqueeze(1)

            # calc hamming distance
            diff = M1_expanded != M2_expanded

            # (# batch, # batch)
            diff1 = diff

            # diff1 = torch.sum(diff, dim=2)
            # assert 0, (diff.shape, diff1.shape, x1.shape, x2.shape)
            def rbf(d, ard):
                if ard:
                    return torch.exp(
                        -torch.sum(d * self.lengthscale, dim=-1)
                        / torch.sum(self.lengthscale)
                    )
                else:
                    return torch.exp(
                        -self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1]
                    )

            def mat52(d, ard):
                raise NotImplementedError

            if exp == "rbf":
                k_cat = rbf(
                    diff1, self.ard_num_dims is not None and self.ard_num_dims > 1
                )
            elif exp == "mat52":
                k_cat = mat52(
                    diff1, self.ard_num_dims is not None and self.ard_num_dims > 1
                )
            else:
                raise ValueError("Exponentiation scheme %s is not recognised!" % exp)
            if diag:
                return torch.diag(k_cat).to(**tkwargs)
            return k_cat.to(**tkwargs)

        else:
            batch_size1, l, n1, m = x1.shape
            batch_size2, _, n2, _ = x2.shape

            assert batch_size2 == batch_size1

            # Expand x1 and x2 to calculate the Hamming distance
            M1_expanded = x1.unsqueeze(3)  # Shape: (batch_size, l, n1, 1, m)
            M2_expanded = x2.unsqueeze(2)  # Shape: (batch_size, l, 1, n2, m)

            # Calculate Hamming distance
            hamming_dist = (
                (M1_expanded != M2_expanded).float().sum(dim=-1)
            )  # Shape: (batch_size, l, n1, n2)

            def rbf(d, ard=False):
                if ard:
                    return torch.exp(-torch.sum(d / self.lengthscale, dim=-1))
                else:
                    return torch.exp(-self.lengthscale * d)

            def mat52(d):
                raise NotImplementedError

            if exp == "rbf":
                k_cat = rbf(hamming_dist)
            elif exp == "mat52":
                k_cat = mat52(hamming_dist)
            else:
                raise ValueError("Exponentiation scheme %s is not recognized!" % exp)

            if diag:
                return torch.diagonal(k_cat, offset=0, dim1=-2, dim2=-1).contiguous()

            return k_cat  # Shape: (batch_size, l, n1, n2)
