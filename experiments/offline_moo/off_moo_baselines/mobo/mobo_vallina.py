import warnings
from copy import deepcopy
from typing import Tuple

import gpytorch
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.models import FixedNoiseGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import (
    ExactMarginalLogLikelihood,
    SumMarginalLogLikelihood,
)
from numpy import ndarray
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from torch import Tensor
from utils import get_N_nondominated_index

from offline_moo.off_moo_baselines.mobo.kernel import (
    OrderKernel,
    TransformedCategorical,
)
from offline_moo.off_moo_baselines.mobo.mobo_utils import tkwargs
from offline_moo.off_moo_baselines.mobo.surrogate_problem import LCB_Problem
from offline_moo.off_moo_bench.collecter import get_operator_dict
from offline_moo.off_moo_bench.task_set import *


class MOBOContinuous:
    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        config: dict,
        solver_kwargs: dict,
        train_gp_data_size: int = 256,
        output_size: int = 256,
        negate=True,
    ) -> None:
        """
        args:
            X_init: '(N,D)' data of decision variable.
            Y_init: '(N,m)' data of objective values.
            ref_point : '(m,)' reference point.
            train_gp_data_size: Size of data for fitting GP.
            bounds: '(2, D)' bounds of decision variable.
            output_size: Size of data for evluating once.
        """
        self.config = config
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]

        assert (
            self.num_obj < 5
        ), "Due to high computational cost, MOBO-qNEHVI is suggested to run on a continuous problem with less than 5 objectives."

        self.X_init, self.Y_init = self._sample_data(
            X_init.detach().cpu().numpy(),
            Y_init.detach().cpu().numpy(),
            train_gp_data_size,
        )
        bounds_ = np.ones((2, self.dim))
        bounds_[0] = solver_kwargs["xl"]
        bounds_[1] = solver_kwargs["xu"]
        self.bounds = torch.from_numpy(bounds_).to(**tkwargs)
        self.ref_point = solver_kwargs["ref_point"].to(**tkwargs)
        self.X_init, self.Y_init = (
            self.X_init.to(**tkwargs),
            self.Y_init.to(**tkwargs),
        )
        if negate:
            self.Y_init *= -1
            self.ref_point *= -1
        self.output_size = output_size

    def _sample_data(self, X_init, Y_init, train_gp_data_size) -> Tuple[Tensor, Tensor]:
        indices_select = get_N_nondominated_index(Y_init, train_gp_data_size)
        Y_init = Y_init[indices_select]
        X_init = X_init[indices_select]
        return torch.tensor(X_init), torch.tensor(Y_init)

    def _get_model(self, train_x, train_obj):
        train_x = normalize(train_x, self.bounds)
        model = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def run(self) -> ndarray:
        """
        return:
             ret: (output_size, D) data decision variable with one BO iteration with output_size batches.
        """
        MC_SAMPLES = 128
        RAW_SAMPLES = 256
        NUM_RESTARTS = 10
        standard_bounds = torch.zeros(2, self.dim, **tkwargs)
        standard_bounds[1] = 1

        model = self._get_model(self.X_init, self.Y_init)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
        with torch.no_grad():
            pred = model.posterior(normalize(self.X_init, self.bounds)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=torch.zeros_like(
                self.ref_point, device=self.ref_point.device, dtype=self.ref_point.dtype
            ),
            Y=pred,
        )
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            X_baseline=normalize(self.X_init, self.bounds),
            partitioning=partitioning,
            sampler=sampler,
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=standard_bounds,
            q=self.output_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 10, "maxiter": 200},
            sequential=True,
        )

        candidates = unnormalize(candidates.detach(), bounds=self.bounds)
        return candidates.cpu().numpy()


class MOBOPermutation:
    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        solver_kwargs: dict,
        config: dict,
        train_gp_data_size: int = 256,
        output_size: int = 256,
        negate=True,
    ) -> None:
        """
        args:
            X_init: '(N,D)' data of decision variable.
            Y_init: '(N,m)' data of objective values.
            ref_point : '(m,)' reference point.
            train_gp_data_size: Size of data for fitting GP.
            bounds: '(2, D)' bounds of decision variable.
            output_size: Size of data for evluating once.
        """
        self.config = config
        X_init = X_init.to(**tkwargs)
        Y_init = Y_init.to(**tkwargs)
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]
        self.X_init, self.Y_init = self._sample_data(
            X_init.detach().cpu().numpy(),
            Y_init.detach().cpu().numpy(),
            train_gp_data_size,
        )
        self.output_size = output_size

    def _sample_data(self, X_init, Y_init, train_gp_data_size) -> Tuple[Tensor, Tensor]:
        indices_select = get_N_nondominated_index(Y_init, train_gp_data_size)
        Y_init = Y_init[indices_select]
        X_init = X_init[indices_select]
        return torch.tensor(X_init), torch.tensor(Y_init)

    def _get_model(self, train_X: Tensor, train_Y: Tensor):
        models = []
        for i in range(train_Y.shape[-1]):
            kernel = OrderKernel().to(**tkwargs)
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

    def run(self) -> ndarray:
        model = self._get_model(self.X_init, self.Y_init)
        problem = LCB_Problem(self.dim, self.num_obj, model)
        print("----nsag2,solving...-----")
        tmp_config = deepcopy(self.config)
        tmp_config.update(
            {
                "sampling": "perm_rnd",
                "crossover": "order",
                "mutation": "inversion",
            }
        )
        operators_dict = get_operator_dict(tmp_config)
        _algo = NSGA2(
            pop_size=self.output_size,
            **operators_dict,
            eliminate_duplicates=True,
        )
        res = minimize(
            problem=problem,
            algorithm=_algo,
            termination=("n_gen", self.config["permutation_n_gen"]),
            verbose=True,
        )
        x = res.pop.get("X")
        return x


class MOBOSequence:
    def __init__(
        self,
        X_init: Tensor,
        Y_init: Tensor,
        config: dict,
        solver_kwargs: dict,
        train_gp_data_size: int = 256,
        output_size: int = 256,
        negate=True,
    ) -> None:
        """
        args:
            X_init: '(N,D)' data of decision variable.
            Y_init: '(N,m)' data of objective values.
            xl:
            xu:
            ref_point : '(m,)' reference point.
            train_gp_data_size: Size of data for fitting GP.
            bounds: '(2, D)' bounds of decision variable.
            output_size: Size of data for evluating once.
        """
        X_init = X_init.to(**tkwargs)
        Y_init = Y_init.to(**tkwargs)
        self.dim = X_init.shape[1]
        self.num_obj = Y_init.shape[1]
        self.X_init, self.Y_init = self._sample_data(
            X_init.detach().cpu().numpy(),
            Y_init.detach().cpu().numpy(),
            train_gp_data_size,
        )
        self.output_size = output_size

        self.xu = solver_kwargs["xu"]
        self.xl = solver_kwargs["xl"]
        self.config = config

    def _sample_data(self, X_init, Y_init, train_gp_data_size) -> Tuple[Tensor, Tensor]:
        indices_select = get_N_nondominated_index(Y_init, train_gp_data_size)
        Y_init = Y_init[indices_select]
        X_init = X_init[indices_select]
        return torch.tensor(X_init).to(**tkwargs), torch.tensor(Y_init).to(**tkwargs)

    def _get_model(self, train_X: Tensor, train_Y: Tensor):
        Y_var = torch.full_like(train_Y, 0.01).to(**tkwargs)
        kernel = TransformedCategorical().to(**tkwargs)
        model = FixedNoiseGP(train_X, train_Y, Y_var, covar_module=kernel).to(**tkwargs)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(**tkwargs)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(**tkwargs)
        fit_gpytorch_mll(mll)
        return model

    def run(self) -> ndarray:
        model = self._get_model(self.X_init, self.Y_init).to(**tkwargs)

        problem = LCB_Problem(self.dim, self.num_obj, model, self.xl, self.xu)
        print("----nsag2,solving...-----")
        tmp_config = deepcopy(self.config)
        tmp_config.update(
            {
                "sampling": "evox_sampling",
                "crossover": "evox_crossover",
                "mutation": "evox_mutation",
                "repair": "evox_repair",
            }
        )
        operator_dict = get_operator_dict(tmp_config)
        _algo = NSGA2(
            pop_size=self.output_size, **operator_dict, eliminate_duplicates=True
        )
        res = minimize(
            problem=problem,
            algorithm=_algo,
            termination=("n_gen", self.config["sequence_n_gen"]),
            verbose=True,
        )

        x = res.pop.get("X")
        return x


class MOBOVallina:
    def __init__(self, config: dict, **kwargs):

        self.config = config
        task_name = config["task"]

        TYPE2SOLVER = {
            "continuous": MOBOContinuous,
            "permutation": MOBOPermutation,
            "sequence": MOBOSequence,
        }

        assert (
            task_name in ALLTASKS
        ), f"task {task_name} not supported in offline-moo-bench"
        if task_name in CONTINUOUSTASKS:
            if task_name in MORL:
                warnings.warn(
                    "MOBO-qNEHVI is not suggested to run on MORL tasks", UserWarning
                )
            self.solver_type = TYPE2SOLVER["continuous"]
        elif task_name in PERMUTATIONTASKS:
            self.solver_type = TYPE2SOLVER["permutation"]
        elif task_name in SEQUENCETASKS:
            self.solver_type = TYPE2SOLVER["sequence"]
        else:
            raise ValueError

        self.solver = self.solver_type(config=config, **kwargs)

    def run(self) -> ndarray:
        return self.solver.run()


if __name__ == "__main__":
    X_init = torch.rand((1000, 7))
    Y_init = torch.rand(1000, 3)
    ref_point = torch.zeros(3)
    train_gp_data_size = 32
    bounds_ = torch.zeros((2, 7))
    bounds_[1] = 1.0
    mobo_once = MOBOVallina(X_init, Y_init, ref_point, train_gp_data_size, bounds_)
    print(mobo_once.run())
