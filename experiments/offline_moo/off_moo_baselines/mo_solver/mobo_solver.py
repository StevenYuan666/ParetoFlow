import os
from typing import Optional

import numpy as np
import torch
from algorithm.mo_solver.base import Solver
from algorithm.mo_solver.external import lhs
from botorch.test_functions.base import MultiObjectiveTestProblem
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
from pymoo.core.problem import Problem
from torch import Tensor
from utils import get_N_nondominated_index

from offline_moo.off_moo_bench.evaluation.metrics import hv

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


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


class MOBOSolver(Solver):
    def __init__(self, n_gen, pop_init_method, batch_size, **kwargs):
        super().__init__(
            n_gen=n_gen, pop_init_method=pop_init_method, batch_size=batch_size
        )
        self.algo_kwargs = kwargs

    def solve(self, problem, X, Y):

        botorch_problem = BotorchProblem(
            problem, ref_point=np.max(Y, axis=0), negate=True
        ).to(**tkwargs)

        X_init, Y_init = self._get_sampling(X, Y, num_samples=100)

        SMOKE_TEST = os.environ.get("SMOKE_TEST")

        from botorch.acquisition.multi_objective.monte_carlo import (
            qExpectedHypervolumeImprovement,
            qNoisyExpectedHypervolumeImprovement,
        )
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.models.gp_regression import FixedNoiseGP
        from botorch.models.model_list_gp_regression import ModelListGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
        from botorch.utils.multi_objective.box_decompositions.non_dominated import (
            FastNondominatedPartitioning,
        )
        from botorch.utils.multi_objective.scalarization import (
            get_chebyshev_scalarization,
        )
        from botorch.utils.sampling import draw_sobol_samples, sample_simplex
        from botorch.utils.transforms import normalize, unnormalize
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

        NOISE_SE = torch.tensor(
            [1e-6 for _ in range(botorch_problem.num_objectives)], **tkwargs
        )

        def initialize_model(train_x, train_obj, state_dict=None):
            # define models for objective and constraint
            train_x = normalize(train_x, botorch_problem.bounds)
            models = []
            for i in range(train_obj.shape[-1]):
                train_y = train_obj[..., i : i + 1]
                train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
                models.append(
                    FixedNoiseGP(
                        train_x, train_y, train_yvar, outcome_transform=Standardize(m=1)
                    )
                )
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            if state_dict is not None:
                model.load_state_dict(state_dict)
            return mll, model

        BATCH_SIZE = 8
        NUM_RESTARTS = 10 if not SMOKE_TEST else 2
        RAW_SAMPLES = 512 if not SMOKE_TEST else 4

        standard_bounds = torch.zeros(2, botorch_problem.dim, **tkwargs)
        standard_bounds[1] = 1

        def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler):
            """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
            # partition non-dominated space into disjoint rectangles
            with torch.no_grad():
                pred = model.posterior(normalize(train_x, botorch_problem.bounds)).mean
            partitioning = FastNondominatedPartitioning(
                ref_point=botorch_problem.ref_point,
                Y=pred,
            )
            acq_func = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=botorch_problem.ref_point,
                partitioning=partitioning,
                sampler=sampler,
            )
            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            # observe new values
            new_x = unnormalize(candidates.detach(), bounds=botorch_problem.bounds)
            new_obj_true = botorch_problem(new_x)
            new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
            return new_x, new_obj, new_obj_true

        def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
            """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
            # partition non-dominated space into disjoint rectangles
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=botorch_problem.ref_point.tolist(),  # use known reference point
                X_baseline=normalize(train_x, botorch_problem.bounds),
                prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
                sampler=sampler,
            )
            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )
            # observe new values
            new_x = unnormalize(candidates.detach(), bounds=botorch_problem.bounds)
            new_obj_true = botorch_problem(new_x)
            new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
            return new_x, new_obj, new_obj_true

        import time
        import warnings

        from botorch import fit_gpytorch_mll
        from botorch.exceptions import BadInitialCandidatesWarning
        from botorch.sampling.normal import SobolQMCNormalSampler
        from botorch.utils.multi_objective.box_decompositions.dominated import (
            DominatedPartitioning,
        )
        from botorch.utils.multi_objective.pareto import is_non_dominated

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        N_BATCH = 20 if not SMOKE_TEST else 10
        MC_SAMPLES = 128 if not SMOKE_TEST else 16

        verbose = True

        # hvs_qehvi = []
        # hvs_qehvi.append(hv(ref_point=botorch_problem.ref_point.cpu().numpy()).do(Y_init))

        hvs_qnehvi = []
        hvs_qnehvi.append(
            hv(nadir_point=botorch_problem.ref_point.cpu().numpy(), y=Y_init)
        )

        # train_x_qehvi, train_obj_qehvi = (
        #     torch.Tensor(X_init).to(**tkwargs),
        #     torch.Tensor(Y_init).to(**tkwargs),
        # )

        train_x_qnehvi, train_obj_qnehvi = (
            torch.Tensor(X_init).to(**tkwargs),
            torch.Tensor(Y_init).to(**tkwargs),
        )

        # mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)

        for iteration in range(1, N_BATCH + 1):
            t0 = time.time()

            # fit_gpytorch_mll(mll_qehvi)
            fit_gpytorch_mll(mll_qnehvi)

            # qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
            qnehvi_sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([MC_SAMPLES])
            )

            # (
            #     new_x_qehvi,
            #     new_obj_qehvi,
            #     new_obj_true_qehvi
            # ) = optimize_qehvi_and_get_observation(
            #     model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler
            # )

            (new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi) = (
                optimize_qnehvi_and_get_observation(
                    model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
                )
            )

            # new_x_qehvi = new_x_qehvi.detach()
            # new_obj_qehvi = new_obj_qehvi.detach()
            # new_obj_true_qehvi = new_obj_true_qehvi.detach()

            new_x_qnehvi = new_x_qnehvi.detach()
            new_obj_qnehvi = new_obj_qnehvi.detach()
            new_obj_true_qnehvi = new_obj_true_qnehvi.detach()

            # assert 0, (new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi)

            # train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            # train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
            # train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])

            train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            # train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

            hvs_qnehvi.append(
                hv(
                    nadir_point=botorch_problem.ref_point.cpu().numpy(),
                    y=train_obj_qnehvi.detach().cpu().numpy(),
                )
            )

            mll_qnehvi, model_qnehvi = initialize_model(
                train_x_qnehvi, train_obj_qnehvi
            )

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume = "
                    f"({hvs_qnehvi[-1]:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.",
                    end="",
                )
            else:
                print(".", end="")

        train_x_qnehvi = train_x_qnehvi.detach().cpu().numpy()
        train_obj_qnehvi = train_obj_qnehvi.detach().cpu().numpy()

        from pymoo.algorithms.moo.nsga2 import NonDominatedSorting

        # fronts = NonDominatedSorting().do(train_obj_qnehvi, return_rank=True, n_stop_if_ranked=256)[0]
        # indices_cnt = 0
        # indices_select = []
        # for front in fronts:
        #     if indices_cnt + len(front) < 256:
        #         indices_cnt += len(front)
        #         indices_select += [int(i) for i in front]
        #     else:
        #         idx = np.random.randint(len(front), size=(256-indices_cnt, ))
        #         indices_select += [int(i) for i in front[idx]]
        #         break
        indices_select = get_N_nondominated_index(train_obj_qnehvi, 256)
        y_sol = train_obj_qnehvi[indices_select]
        x_sol = train_x_qnehvi[indices_select]

        return {"x": x_sol, "y": y_sol}

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
