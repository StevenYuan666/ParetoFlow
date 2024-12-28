import os

import numpy as np
import torch
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair

from offline_moo.off_moo_bench.problem.base import BaseProblem

from .MOTSProblemDef import augment_xy_data_by_64_fold_2obj, get_random_problems


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


class MOTSP(BaseProblem):
    def __init__(self, n_obj=2, problem_size=500):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="comb. opt",
            n_dim=problem_size,
            n_obj=n_obj,
            xl=0,
            xu=problem_size - 1,
        )
        self.problem_size = problem_size
        self.problem_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"MOTSP_problem_{problem_size}.pt",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_problems()

    def load_problems(self, aug_factor=1, problems=None):
        if problems is not None:
            self.problems = problems
        elif os.path.exists(self.problem_file):
            self.problems = torch.load(f=self.problem_file)
        else:
            self.problems = get_random_problems(1, self.problem_size)
            torch.save(obj=self.problems, f=self.problem_file)

        # problems.shape: (1, problem, 2)
        if aug_factor > 1:
            if aug_factor == 64:
                self.problems = augment_xy_data_by_64_fold_2obj(self.problems)
            else:
                raise NotImplementedError

        self.problems = self.problems.to(self.device)

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).reshape((x.shape[0], 1, -1)).to(self.device)
        self.batch_size = x.shape[0]

        expanded_problems = self.problems.repeat(self.batch_size, 1, 1)

        gathering_index = x.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 4
        )
        # shape: (batch, 1, problem, 4)
        seq_expanded = expanded_problems[:, None, :, :].expand(
            self.batch_size, 1, self.problem_size, 4
        )

        # assert 0, gathering_index
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, q, problem, 4)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths_obj1 = (
            ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
        )
        segment_lengths_obj2 = (
            ((ordered_seq[:, :, :, 2:] - rolled_seq[:, :, :, 2:]) ** 2).sum(3).sqrt()
        )

        travel_distances_obj1 = segment_lengths_obj1.sum(2)
        travel_distances_obj2 = segment_lengths_obj2.sum(2)

        travel_distances_vec = torch.stack(
            [travel_distances_obj1, travel_distances_obj2], axis=2
        ).reshape((self.batch_size, self.n_obj))

        # out["G"] = np.ones(self.batch_size)
        # for i, x_i in enumerate(x):
        #     if torch.equal(x_i.data.sort(1)[0], \
        #             torch.arange(x_i.size(1), out=x_i.data.new()).view(1, -1).expand_as(x_i)):
        #         out["G"][i] = -1
        out["F"] = travel_distances_vec.cpu().numpy()
        # shape: (batch, pomo)


class BiTSP500(MOTSP):
    def __init__(self):
        super().__init__(problem_size=500)

    def get_nadir_point(self):
        return np.array([236.25364685, 230.09597778])

    def get_ideal_point(self):
        return np.array([46.97628403, 46.6463623])


class BiTSP100(MOTSP):
    def __init__(self):
        super().__init__(problem_size=100)

    def get_nadir_point(self):
        return np.array([48.83366013, 52.39511108])

    def get_ideal_point(self):
        return np.array([8.31447029, 8.56387234])


class BiTSP50(MOTSP):
    def __init__(self):
        super().__init__(problem_size=50)

    def get_nadir_point(self):
        return np.array([24.1007843, 26.50193977])

    def get_ideal_point(self):
        return np.array([5.74934673, 6.11038303])


class BiTSP20(MOTSP):
    def __init__(self):
        super().__init__(problem_size=20)

    def get_nadir_point(self):
        return np.array([12.23731232, 11.58497334])

    def get_ideal_point(self):
        return np.array([4.0333252, 3.95475602])
