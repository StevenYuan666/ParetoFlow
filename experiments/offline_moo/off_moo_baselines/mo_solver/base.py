import numpy as np
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from offline_moo.off_moo_baselines.mo_solver.external import lhs


class Solver:
    """
    Multi-objective solver
    """

    def __init__(self, n_gen, pop_init_method, batch_size):
        """
        Input:
            n_gen: number of generations to solve
            pop_init_method: method to initialize population
            algo: class of multi-objective algorithm to use
            kwargs: other keyword arguments for algorithm to initialize
        """
        self.n_gen = n_gen
        self.pop_init_method = pop_init_method
        self.batch_size = batch_size
        self.solution = None

    def solve(self, problem, X, Y):
        """
        Solve the multi-objective problem
        """
        raise NotImplementedError

    def _get_sampling(self, X, Y):
        """
        Initialize population from data
        """
        if self.pop_init_method == "lhs":
            sampling = LatinHypercubeSampling()
        elif self.pop_init_method == "nds":
            sorted_indices = NonDominatedSorting().do(Y)
            pop_size = self.pop_size
            sampling = X[np.concatenate(sorted_indices)][:pop_size]
            # NOTE: use lhs if current samples are not enough
            if len(sampling) < pop_size:
                rest_sampling = lhs(X.shape[1], pop_size - len(sampling))
                sampling = np.vstack([sampling, rest_sampling])
        elif self.pop_init_method == "random":
            sampling = FloatRandomSampling()
        else:
            raise NotImplementedError

        return sampling
