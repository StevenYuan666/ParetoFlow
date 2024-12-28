from offline_moo.off_moo_bench.problem.moo_molecule_funcs.properties import (
    SUPPORTED_PROPERTIES,
    MOOMoleculeFunction,
)

f = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))

import numpy as np
import torch

from offline_moo.off_moo_bench.problem.base import BaseProblem


class Molecule(BaseProblem):
    def __init__(self, f=f):
        super().__init__(
            name=self.__class__.__name__,
            n_dim=f.dim,
            n_obj=f.num_objectives,
            problem_type="continuous",
            xl=f.bounds[0].cpu().numpy(),
            xu=f.bounds[1].cpu().numpy(),
            nadir_point=np.array([0.0, 0.0]),
            ideal_point=np.array([-1.0, -1.0]),
            # n_constr = 1
        )
        self.f = f
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.last_pop = None

    def _evaluate(self, x, out, mode="eval", *args, **kwargs):
        assert mode in ["eval", "collect"]
        x = torch.from_numpy(x).to(self.device)
        if mode == "collect":
            res = self.f(x).detach().cpu().numpy()
        else:
            res = []
            feasible = []
            for i, x_i in enumerate(x):
                try:
                    res_i = self.f(x_i.reshape(1, -1)).detach().cpu().numpy()
                    res.append(res_i)
                    feasible.append(1)
                except:
                    res.append(np.array([-1.0, -1.0]).reshape((1, 2)))
                    feasible.append(0)
            res = np.concatenate(res, axis=0)
            feasible = np.array(feasible)

        out["F"] = res * (-1)
        if mode == "eval":
            out["feasible"] = feasible

    def get_nadir_point(self):
        return np.array([-0.0, -0.0])

    def get_ideal_point(self):
        return np.array([-0.95, -0.36])
