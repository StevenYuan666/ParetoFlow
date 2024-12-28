import numpy as np
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from offline_moo.off_moo_baselines.mo_solver.base import Solver
from offline_moo.off_moo_baselines.mo_solver.external import lhs


class MOEASolver(Solver):
    """
    Multi-objective solver
    """

    def __init__(
        self, n_gen, pop_init_method, batch_size, algo, callback=None, **kwargs
    ):
        """
        Input:
            n_gen: number of generations to solve
            pop_init_method: method to initialize population
            algo: class of multi-objective algorithm to use
            kwargs: other keyword arguments for algorithm to initialize
        """
        super().__init__(
            n_gen=n_gen, batch_size=batch_size, pop_init_method=pop_init_method
        )
        self.algo_type = algo
        self.callback = callback
        self.algo_kwargs = kwargs

        if "pop_size" not in self.algo_kwargs.keys():
            assert "ref_dirs" in self.algo_kwargs.keys()
            self.pop_size = len(self.algo_kwargs["ref_dirs"])
        else:
            self.pop_size = self.algo_kwargs["pop_size"]

    def solve(self, problem, X, Y):
        """
        Solve the multi-objective problem
        """
        # initialize population
        sampling = self._get_sampling(X, Y)
        # setup algorithm
        algo = self.algo_type(sampling=sampling, **self.algo_kwargs)
        if self.callback is not None:
            try:
                algo.callback = self.callback._do
            except:
                algo.callback = self.callback

        # optimization
        res = minimize(problem, algo, ("n_gen", self.n_gen))

        # construct solution
        self.solution = {
            "x": res.pop.get("X"),
            "y": res.pop.get("F"),
            "algo": res.algorithm,
        }

        # fill the solution in case less than batch size
        pop_size = len(self.solution["x"])
        if pop_size < self.batch_size:
            indices = np.concatenate(
                [
                    np.arange(pop_size),
                    np.random.choice(np.arange(pop_size), self.batch_size - pop_size),
                ]
            )
            self.solution["x"] = np.array(self.solution["x"])[indices]
            self.solution["y"] = np.array(self.solution["y"])[indices]

        # self.callback.plot_all()

        return self.solution
