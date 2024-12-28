from typing import Tuple

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition,
    random_search_optimizer,
    sample_optimal_points,
)
from botorch.models.gp_regression import SingleTaskGP
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from numpy import ndarray
from torch import Tensor
from utils import get_N_nondominated_index

from offline_moo.off_moo_baselines.mobo.mobo_utils import tkwargs
from offline_moo.off_moo_bench.task_set import *


class MOBOJESContinuous:
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
            self.num_obj < 4
        ), "Due to high computational cost, MOBO-JES is suggested to run on a continuous problem with less than 4 objectives."

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
        RAW_SAMPLES = 256
        NUM_RESTARTS = 10
        standard_bounds = torch.zeros(2, self.dim, **tkwargs)
        standard_bounds[1] = 1
        model = self._get_model(self.X_init, self.Y_init)
        num_pareto_samples = 10
        num_pareto_points = 10

        # We set the parameters for the random search
        optimizer_kwargs = {
            "pop_size": 5000,
            "max_tries": 50,
        }

        ps, pf = sample_optimal_points(
            model=model,
            bounds=standard_bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
            optimizer=random_search_optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        hypercell_bounds = compute_sample_box_decomposition(pf)

        jes_lb = qLowerBoundMultiObjectiveJointEntropySearch(
            model=model,
            pareto_sets=ps,
            pareto_fronts=pf,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
        )

        candidates, _ = optimize_acqf(
            acq_function=jes_lb,
            bounds=standard_bounds,
            q=self.output_size,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            options={"batch_limit": 10, "maxiter": 200},
            sequential=True,
        )
        candidates = unnormalize(candidates.detach(), bounds=self.bounds)
        return candidates.cpu().numpy()


class MOBOJES:
    def __init__(self, config: dict, **kwargs):
        self.config = config
        task_name = config["task"]

        TYPE2SOLVER = {
            "continuous": MOBOJESContinuous,
        }

        assert (
            task_name in ALLTASKS
        ), f"task {task_name} not supported in offline-moo-bench"
        if task_name in CONTINUOUSTASKS:
            if task_name in MORL:
                raise ValueError("MOBO-JES is not suggested to run on MORL tasks")
            self.solver_type = TYPE2SOLVER["continuous"]
        elif task_name in PERMUTATIONTASKS:
            raise ValueError(
                "MOBO-JES is not supported to run on permutation-based tasks"
            )
        elif task_name in SEQUENCETASKS:
            raise ValueError("MOBO-JES is not supported to run on sequence-based tasks")
        else:
            raise ValueError

        self.solver = self.solver_type(config=config, **kwargs)

    def run(self) -> ndarray:
        return self.solver.run()
