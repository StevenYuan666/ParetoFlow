import os

import numpy as np
import torch
from pymoo.core.repair import Repair

# from pymoo.core.problem import Problem
from offline_moo.off_moo_bench.problem.base import BaseProblem

from .MOTSProblemDef_3obj import augment_xy_data_by_n_fold_3obj, get_random_problems


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


class MOTSP_3obj(BaseProblem):
    def __init__(self, n_obj=3, problem_size=500):
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
            f"MOTSP3obj_problem_{problem_size}.pt",
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
            if aug_factor <= 512:
                self.batch_size = self.batch_size * aug_factor
                self.problems = augment_xy_data_by_n_fold_3obj(
                    self.problems, aug_factor
                )
            else:
                raise NotImplementedError

        self.problems = self.problems.to(self.device)

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.from_numpy(x).reshape((x.shape[0], 1, -1)).to(self.device)
        self.batch_size = x.shape[0]

        expanded_problems = self.problems.repeat(self.batch_size, 1, 1)

        gathering_index = x.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 6
        )
        # shape: (batch, 1, problem, 4)
        seq_expanded = expanded_problems[:, None, :, :].expand(
            self.batch_size, 1, self.problem_size, 6
        )

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, q, problem, 4)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

        segment_lengths_obj1 = (
            ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
        )
        segment_lengths_obj2 = (
            ((ordered_seq[:, :, :, 2:] - rolled_seq[:, :, :, 2:]) ** 2).sum(3).sqrt()
        )
        segment_lengths_obj3 = (
            ((ordered_seq[:, :, :, 4:] - rolled_seq[:, :, :, 4:]) ** 2).sum(3).sqrt()
        )

        travel_distances_obj1 = segment_lengths_obj1.sum(2)
        travel_distances_obj2 = segment_lengths_obj2.sum(2)
        travel_distances_obj3 = segment_lengths_obj3.sum(2)

        travel_distances_vec = torch.stack(
            [travel_distances_obj1, travel_distances_obj2, travel_distances_obj3],
            axis=2,
        ).reshape((self.batch_size, self.n_obj))

        # out["G"] = np.ones(self.batch_size)
        # for i, x_i in enumerate(x):
        #     if torch.equal(x_i.data.sort(1)[0], \
        #             torch.arange(x_i.size(1), out=x_i.data.new()).view(1, -1).expand_as(x_i)):
        #         out["G"][i] = -1
        out["F"] = travel_distances_vec.cpu().numpy()


class TriTSP100(MOTSP_3obj):
    def __init__(self):
        super().__init__(problem_size=100)

    def get_nadir_point(self):
        return np.array([51.48603821, 76.25702667, 50.1726265])

    def get_ideal_point(self):
        return np.array([9.24660873, 30.9209404, 10.63162804])


class TriTSP50(MOTSP_3obj):
    def __init__(self):
        super().__init__(problem_size=50)

    def get_nadir_point(self):
        return np.array([28.13719368, 39.39593506, 26.49636459])

    def get_ideal_point(self):
        return np.array([5.90725708, 16.76548958, 5.73812628])


class TriTSP20(MOTSP_3obj):
    def __init__(self):
        super().__init__(problem_size=20)

    def get_nadir_point(self):
        return np.array([11.38083839, 16.21029663, 10.02484703])

    def get_ideal_point(self):
        return np.array([3.94876862, 8.32251167, 3.68904114])


if __name__ == "__main__":
    problem = MOTSP_3obj()
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.ox import OrderCrossover
    from pymoo.operators.mutation.inversion import InversionMutation
    from pymoo.operators.sampling.rnd import PermutationRandomSampling
    from pymoo.optimize import minimize

    x_all = []
    y_all = []

    def callback(algorithm):
        x = algorithm.pop.get("X")
        y = algorithm.pop.get("F")
        x_all.append(x)
        y_all.append(y)

    algorithm = NSGA2(
        pop_size=256,
        sampling=PermutationRandomSampling(),
        mutation=InversionMutation(),
        crossover=OrderCrossover(),
        repair=StartFromZeroRepair(),
        eliminate_duplicates=True,
        callback=callback,
    )
    res = minimize(
        problem=problem, algorithm=algorithm, termination=("n_gen", 10000), verbose=True
    )
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_all[0][:, 0], y_all[0][:, 1], color="pink")
    plt.scatter(y_all[100][:, 0], y_all[100][:, 1], color="blue")
    plt.scatter(y_all[500][:, 0], y_all[500][:, 1], color="orange")
    plt.scatter(y_all[1000][:, 0], y_all[1000][:, 1], color="yellow")
    plt.scatter(y_all[1500][:, 0], y_all[1500][:, 1], color="pink")
    plt.scatter(y_all[2000][:, 0], y_all[2000][:, 1], color="blue")
    plt.scatter(y_all[2500][:, 0], y_all[2500][:, 1], color="orange")

    plt.scatter(y_all[3000][:, 0], y_all[3000][:, 1], color="pink")
    plt.scatter(y_all[3500][:, 0], y_all[3500][:, 1], color="blue")
    plt.scatter(y_all[4000][:, 0], y_all[4000][:, 1], color="orange")
    plt.scatter(y_all[4500][:, 0], y_all[4500][:, 1], color="yellow")
    plt.scatter(y_all[5000][:, 0], y_all[5000][:, 1], color="pink")
    plt.scatter(y_all[5500][:, 0], y_all[5500][:, 1], color="blue")
    plt.scatter(y_all[6000][:, 0], y_all[6000][:, 1], color="orange")
    plt.scatter(y_all[6500][:, 0], y_all[6500][:, 1], color="pink")
    plt.scatter(y_all[7000][:, 0], y_all[7000][:, 1], color="blue")
    plt.scatter(y_all[7500][:, 0], y_all[7500][:, 1], color="orange")
    plt.scatter(y_all[8000][:, 0], y_all[8000][:, 1], color="yellow")
    plt.scatter(y_all[8500][:, 0], y_all[8500][:, 1], color="pink")
    plt.scatter(y_all[9000][:, 0], y_all[9000][:, 1], color="blue")
    plt.scatter(y_all[9500][:, 0], y_all[9500][:, 1], color="orange")
    # plt.scatter(y_all[10000][:, 0], y_all[10000][:, 1], color='pink')
    # plt.scatter(y_all[30000][:, 0], y_all[30000][:, 1], color='blue')
    # plt.scatter(y_all[50000][:, 0], y_all[50000][:, 1], color='orange')
    # plt.scatter(y_all[70000][:, 0], y_all[70000][:, 1], color='yellow')
    # plt.scatter(y_all[90000][:, 0], y_all[90000][:, 1], color='pink')
    plt.scatter(y_all[-1][:, 0], y_all[-1][:, 1], color="blue")
    plt.scatter(res.F[:, 0], res.F[:, 1], color="red")
    plt.savefig("test2.png")
    print(res.CV)

    np.save(arr=np.concatenate(y_all, axis=0), file="y_all_tsp.npy")
    np.save(arr=np.concatenate(x_all, axis=0), file="x_all_tsp.npy")

    y_initial = y_all[0]
    print(problem.evaluate(x_all[-1])[0])
