import numpy as np
import torch
from pymoo.core.problem import Problem

from offline_moo.off_moo_baselines.mobo.mobo_utils import tkwargs


class LCB_Problem(Problem):
    def __init__(self, n_var, n_obj, model, xl=None, xu=None):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
        )
        self.model = model

    def _get_acq_value(self, X, model):
        X = torch.tensor(X)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean - 0.2 * (torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._get_acq_value(x, self.model)


class AcqfProblem(Problem):
    def __init__(self, n_var, acq_func, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.acq_func = acq_func

    def _evaluate(self, x, out, *args, **kwargs):
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x).to(**tkwargs)
        out["F"] = self.acq_func(x.unsqueeze(1)).reshape(
            -1, 1
        ).detach().cpu().numpy() * (-1)
