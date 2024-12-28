import math
from typing import Optional

import gpytorch
import numpy as np
import torch
from algorithm.mo_solver.base import Solver
from algorithm.mo_solver.external import lhs
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.test_functions.base import MultiObjectiveTestProblem
from gpytorch.kernels import Kernel
from numpy import ndarray
from pymoo.algorithms.moo.nsga2 import NSGA2, NonDominatedSorting
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from torch import Tensor

from offline_moo.off_moo_bench.evaluation.metrics import hv

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        """

        Returns
        -------
        object
        """
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)


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
        M1_expanded = x1.unsqueeze(1)
        M2_expanded = x2.unsqueeze(0)

        diff = M1_expanded != M2_expanded

        diff1 = diff

        # diff1 = torch.sum(diff, dim=2)
        def rbf(d, ard):
            if ard:
                return torch.exp(
                    -torch.sum(d * self.lengthscale, dim=-1)
                    / torch.sum(self.lengthscale)
                )
            else:
                return torch.exp(-self.lengthscale * torch.sum(d, dim=-1) / x1.shape[1])

        def mat52(d, ard):
            raise NotImplementedError

        if exp == "rbf":
            k_cat = rbf(diff1, self.ard_num_dims is not None and self.ard_num_dims > 1)
        elif exp == "mat52":
            k_cat = mat52(
                diff1, self.ard_num_dims is not None and self.ard_num_dims > 1
            )
        else:
            raise ValueError("Exponentiation scheme %s is not recognised!" % exp)
        if diag:
            return torch.diag(k_cat).to(**tkwargs)
        return k_cat.to(**tkwargs)


class FeatureCache:
    def __init__(self, input_type="torch"):
        self.input_type = input_type
        self.cache = dict()

    def _get_key(self, x):
        return tuple(x.tolist())

    def push(self, x):
        feature = self.get(x)
        if feature is None:
            feature = self._featurize(x, self.input_type)
            self.cache[self._get_key(x)] = feature
        return feature

    def get(self, x):
        return self.cache.get(self._get_key(x), None)

    def _featurize(self, x, ret_type="torch"):
        assert ret_type in ["torch", "numpy"]
        if ret_type == "torch":
            assert x.dim() == 1
        else:
            assert x.ndim == 1
        featurize_x = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                featurize_x.append(1 if x[i] > x[j] else -1)
        if ret_type == "torch":
            featurize_x = torch.tensor(featurize_x, dtype=torch.float)
        elif ret_type == "numpy":
            featurize_x = np.array(featurize_x, dtype=np.float64)
        else:
            assert 0
        normalizer = np.sqrt(len(x) * (len(x) - 1) / 2)
        return featurize_x / normalizer


import gpytorch
from botorch import fit_gpytorch_mll
from botorch.models import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import Kernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from numpy import ndarray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize


class StartFromZeroRepair(Repair):
    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


feature_cache = FeatureCache()


class OrderKernel(Kernel):
    has_lengthscale = True
    """
    def _count_discordant_pairs(self, x1, x2):
        count = 0
        n = len(x1)
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the pair is discordant
                if (x1[i] < x1[j] and x2[i] > x2[j]) or (x1[i] > x1[j] and x2[i] < x2[j]):
                    count += 1
        return count
    """

    def forward(self, X, X2, **params):
        mat = torch.zeros((len(X), len(X2))).to(X)
        x1 = []
        for i in range(len(X)):
            x1.append(feature_cache.push(X[i]))
        x2 = []
        for j in range(len(X2)):
            x2.append(feature_cache.push(X2[j]))
        # mat = self._count_discordant_pairs(x1, x2)
        x1 = torch.vstack(x1).to(**tkwargs)
        x2 = torch.vstack(x2).to(**tkwargs)
        x1 = torch.reshape(x1, (x1.shape[0], 1, -1))
        x2 = torch.reshape(x2, (1, x2.shape[0], -1))
        x1 = torch.tile(x1, (1, x2.shape[0], 1))
        x2 = torch.tile(x2, (x1.shape[0], 1, 1))
        mat = torch.sum((x1 - x2) ** 2, dim=-1)
        mat = torch.exp(-self.lengthscale * mat)
        return mat


class LCB_Problem(Problem):
    def __init__(self, n_var, n_obj, model, xl=None, xu=None):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.model = model

    def get_acq_value(self, X, model):
        X = torch.tensor(X).to(**tkwargs)
        with torch.no_grad():
            X = X.to(**tkwargs)
            posterior = model.posterior(X)
            mean = posterior.mean
            var = posterior.variance
            return (mean - 0.2 * (torch.sqrt(var))).detach().cpu().numpy()

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.get_acq_value(x, self.model)


class BotorchProblem(MultiObjectiveTestProblem):
    def __init__(
        self,
        pymoo_problem: Problem,
        ref_point: np.ndarray,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        self.dim = pymoo_problem.n_var
        self.num_objectives = pymoo_problem.n_obj
        self._bounds = list(zip(pymoo_problem.xl, pymoo_problem.xu))
        self._ref_point = list(ref_point)
        super().__init__(noise_std=noise_std, negate=negate)
        self.model = pymoo_problem.model

    def evaluate_true(self, X: Tensor) -> Tensor:

        if isinstance(self.model, list):
            assert len(self.model) == self.n_obj
            y = torch.zeros((0, 1)).to(self.device)
            for model in self.model:
                res = model(X)
                y = torch.cat((y, res), axis=0)
        else:
            try:
                y = self.model(X)
            except:
                y = self.model(X, forward_objs=list(self.model.obj2head.keys()))
        return y


class MOBOSolver_seq_perm(Solver):
    def __init__(
        self,
        n_gen,
        pop_init_method,
        batch_size,
        env_name,
        xl=None,
        xu=None,
        var_type="permutation",
        **kwargs
    ):
        super().__init__(
            n_gen=n_gen, pop_init_method=pop_init_method, batch_size=batch_size
        )
        self.algo_kwargs = kwargs
        self.var_type = var_type  #  "permutation"/"sequence"
        self.xl = xl
        self.xu = xu
        self.env_name = env_name

    def _get_sampling(self, X, Y, num_samples):
        if self.pop_init_method == "nds":
            sorted_indices = NonDominatedSorting().do(Y)
            sampling = (
                X[np.concatenate(sorted_indices)][:num_samples],
                Y[np.concatenate(sorted_indices)][:num_samples],
            )
            # NOTE: use lhs if current samples are not enough
            if len(sampling[0]) < num_samples:
                rest_sampling_X = lhs(X.shape[1], num_samples - len(sampling[0]))
                rest_sampling_Y = lhs(Y.shape[1], num_samples - len(sampling[0]))
                sampling = (
                    np.vstack([sampling[0], rest_sampling_X]),
                    np.vstack([sampling[1], rest_sampling_Y]),
                )
        else:
            raise NotImplementedError

        return sampling

    def _get_model(self, train_X: Tensor, train_Y: Tensor):
        models = []
        for i in range(train_Y.shape[-1]):
            if self.var_type == "permutation":
                kernel = OrderKernel().to(**tkwargs)
            else:
                kernel = TransformedCategorical().to(**tkwargs)
            train_y = train_Y[..., i : i + 1]
            train_yvar = torch.full_like(train_y, 0.01**2)
            models.append(
                FixedNoiseGP(train_X, train_y, train_yvar, covar_module=kernel)
            )
        model = ModelListGP(*models)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
        mll = SumMarginalLogLikelihood(likelihood, model).to(**tkwargs)
        fit_gpytorch_mll(mll)
        return model

    def solve(self, problem, X, Y):
        botorch_problem = BotorchProblem(
            problem, ref_point=np.max(Y, axis=0), negate=False
        ).to(**tkwargs)
        X_train, Y_train = self._get_sampling(X, Y, num_samples=128)

        if isinstance(X_train, np.ndarray):
            X_train = torch.from_numpy(X_train).to(**tkwargs)
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.from_numpy(Y_train).to(**tkwargs)

        BATCH_SIZE = 4
        N_BATCH = 32
        POP_SIZE = 32
        N_GEN = 50
        for iteration in range(1, N_BATCH + 1):
            model = self._get_model(X_train, Y_train)
            if self.var_type == "permutation":
                problem = LCB_Problem(
                    botorch_problem.dim, botorch_problem.num_objectives, model
                )
            else:
                problem = LCB_Problem(
                    botorch_problem.dim,
                    botorch_problem.num_objectives,
                    model,
                    xl=self.xl,
                    xu=self.xu,
                )
            if self.var_type == "permutation":
                _algo = NSGA2(
                    pop_size=POP_SIZE,
                    sampling=PermutationRandomSampling(),
                    mutation=InversionMutation(),
                    crossover=OrderCrossover(),
                    repair=(
                        StartFromZeroRepair()
                        if self.env_name in ["mo_tsp", "mo_cvrp"]
                        else None
                    ),
                    eliminate_duplicates=True,
                )
            else:
                _algo = NSGA2(
                    pop_size=POP_SIZE,
                    sampling=get_sampling("int_random"),
                    crossover=get_crossover(name="int_sbx", prob=1.0, eta=3.0),
                    mutation=get_mutation(name="int_pm", prob=1.0, eta=3.0),
                    eliminate_duplicates=True,
                )
            res = minimize(
                problem=problem, algorithm=_algo, termination=("n_gen", N_GEN)
            )
            x = res.pop.get("X")
            """
            Choose 4 solutions from the 32 candidates to maximize LCB-HVI
            """
            ref_point = (
                botorch_problem.ref_point.detach()
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )
            flag = [False] * POP_SIZE
            next_indices = []
            y_now = problem.get_acq_value(X_train.detach().cpu().numpy(), model)
            hv_indicator = Hypervolume(ref_point=ref_point)
            for k in range(BATCH_SIZE):
                max_hvc = 0.0
                max_hvc_idx = 0
                max_hvc_obj = problem.get_acq_value(x[0].reshape(1, -1), model).reshape(
                    1, -1
                )
                for j in range(POP_SIZE):
                    if flag[j]:
                        # If we has chosen this point
                        continue
                    acq_value_j = problem.get_acq_value(
                        x[j].reshape(1, -1), model
                    ).reshape(1, -1)
                    hvc = hv_indicator.do(
                        np.concatenate((y_now, acq_value_j), axis=0)
                    ) - hv_indicator.do(y_now)
                    # Calc the LCB-HVC of this point
                    if hvc >= max_hvc:
                        max_hvc_idx = j
                        max_hvc_obj = acq_value_j
                next_indices.append(max_hvc_idx)
                flag[max_hvc_idx] = True
                y_now = np.concatenate((y_now, max_hvc_obj), axis=0)
            new_x = torch.tensor(x[next_indices]).to(**tkwargs).detach()
            new_obj = botorch_problem(new_x).to(**tkwargs).detach()

            X_train = torch.cat((X_train, new_x), dim=0)
            Y_train = torch.cat((Y_train, new_obj), dim=0)
        X_train = X_train.detach().cpu().numpy().astype(np.int64)
        Y_train = Y_train.detach().cpu().numpy().astype(np.int64)
        return {"x": X_train, "y": Y_train}
