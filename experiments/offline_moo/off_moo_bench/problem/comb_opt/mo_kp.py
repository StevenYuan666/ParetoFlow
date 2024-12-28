import os

import numpy as np
import torch
from pymoo.core.problem import Problem

from offline_moo.off_moo_bench.problem.base import BaseProblem


class MOKP(BaseProblem):
    def __init__(self, n_obj=2, problem_size=200):
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
            f"MOKP_problem_{problem_size}.pt",
        )
        self.pomo_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_problems()

    def load_problems(self, aug_factor=1, problems=None):

        if problems is not None:
            self.problems = problems
        elif os.path.exists(self.problem_file):
            self.problems = torch.load(self.problem_file)
        else:
            from .MOKProblemDef import get_random_problems

            self.problems = get_random_problems(1, self.problem_size)
            torch.save(f=self.problem_file, obj=self.problems)

        # problems.shape: (batch, problem, 2)
        if aug_factor > 1:
            raise NotImplementedError

        self.problems = self.problems.to(self.device)

    def _evaluate(self, x, out, *args, **kwargs):
        # Status
        ####################################

        x = torch.from_numpy(x).reshape(x.shape[0], 1, -1).to(self.device)
        self.batch_size = x.shape[0]

        if self.problems.shape[0] == 1:
            self.problems = self.problems.repeat(self.batch_size, 1, 1)

        # items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size+1, 3)
        # gathering_index = x[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 3)
        # selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)

        self.items_and_a_dummy = torch.Tensor(
            np.zeros((self.batch_size, self.problem_size + 1, 3))
        ).to(self.device)
        self.items_and_a_dummy[:, : self.problem_size, :] = self.problems
        self.item_data = self.items_and_a_dummy[:, : self.problem_size, :]

        if self.problem_size == 50:
            capacity = 12.5
        elif self.problem_size == 100:
            capacity = 25
        elif self.problem_size == 200:
            capacity = 25
        else:
            raise NotImplementedError
        self.capacity = (torch.Tensor(np.ones((self.batch_size,))) * capacity).to(
            self.device
        )

        self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size,))).to(
            self.device
        )
        self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size,))).to(
            self.device
        )

        self.finished = torch.BoolTensor(np.zeros((self.batch_size,))).to(self.device)

        for i in range(x.shape[2]):
            selected_item = x[:, :, i].reshape((-1,))
            # selected_item_data = torch.gather(self.item_data, 1, selected_item)
            selected_item_data = torch.stack(
                [self.item_data[i, selected_item[i], :] for i in range(self.batch_size)]
            ).to(self.device)

            if self.finished.all():
                break

            exceed_capacity = self.capacity - selected_item_data[:, 0] < 0
            self.finished[exceed_capacity] = True
            unfinished_batches = ~self.finished

            self.accumulated_value_obj1[unfinished_batches] += selected_item_data[
                unfinished_batches, 1
            ]
            self.accumulated_value_obj2[unfinished_batches] += selected_item_data[
                unfinished_batches, 2
            ]
            self.capacity[unfinished_batches] -= selected_item_data[
                unfinished_batches, 0
            ]

        res = torch.stack(
            [self.accumulated_value_obj1, self.accumulated_value_obj2], axis=1
        ) * (-1)

        out["F"] = res.cpu().numpy()


class BiKP200(MOKP):
    def __init__(self):
        super().__init__(problem_size=200)

    def get_nadir_point(self):
        return np.array([-11.99687481, -12.96243858])

    def get_ideal_point(self):
        return np.array([-53.51074982, -52.68140411])


class BiKP100(MOKP):
    def __init__(self):
        super().__init__(problem_size=100)

    def get_nadir_point(self):
        return np.array([-9.92121315, -11.35636044])

    def get_ideal_point(self):
        return np.array([-42.59313202, -37.79801178])


class BiKP50(MOKP):
    def __init__(self):
        super().__init__(problem_size=50)

    def get_nadir_point(self):
        return np.array([-3.46616459, -4.13344097])

    def get_ideal_point(self):
        return np.array([-17.50688744, -14.69023991])
