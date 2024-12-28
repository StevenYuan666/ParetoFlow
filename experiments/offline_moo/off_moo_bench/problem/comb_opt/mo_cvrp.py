import os

import numpy as np
import torch
from pymoo.core.repair import Repair

from offline_moo.off_moo_bench.problem.base import BaseProblem

from .MOCVRProblemDef import augment_xy_data_by_8_fold, get_random_problems


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X


class MOCVRP(BaseProblem):
    def __init__(self, n_obj=2, problem_size=100):
        if problem_size == 20:
            ref = np.array([30, 4])  # 20
        elif problem_size == 50:
            ref = np.array([45, 4])
        elif problem_size == 100:
            ref = np.array([80, 4])
        else:
            return NotImplementedError
        super().__init__(
            name=self.__class__.__name__,
            problem_type="comb. opt",
            n_dim=problem_size + 1,
            n_obj=n_obj,
            xl=0,
            xu=problem_size,
            nadir_point=ref,
            ideal_point=[0, 0],
        )
        self.problem_size = problem_size
        self.problem_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"MOCVRP_problem_{problem_size}.pt",
        )
        self.pomo_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_problems()

    def load_problems(self, aug_factor=1, problems=None):

        if problems is not None:
            depot_xy, node_xy, node_demand = problems
        elif os.path.exists(self.problem_file):
            depot_xy, node_xy, node_demand = torch.load(self.problem_file).values()
        else:
            depot_xy, node_xy, node_demand = get_random_problems(1, self.problem_size)
            problems = {
                "depot_xy": depot_xy,
                "node_xy": node_xy,
                "node_demand": node_demand,
            }
            torch.save(obj=problems, f=self.problem_file)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1).to(self.device)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(1, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1).to(
            self.device
        )
        # shape: (batch, problem+1)

    def _evaluate(self, x, out, *args, **kwargs):

        x = torch.from_numpy(x).reshape((x.shape[0], 1, -1)).to(self.device)

        repaired_x = []

        for i, x_i in enumerate(x):
            load = 1
            _copy_x_i = x_i.clone()
            cum_insertion = 0
            for j, x_ij in enumerate(x_i.flatten()):
                if load - self.depot_node_demand[0, j] < 0:
                    inserted_zero = torch.zeros((1, 1), dtype=torch.int64).to(
                        self.device
                    )
                    _copy_x_i = torch.cat(
                        (
                            _copy_x_i[:, : cum_insertion + j],
                            inserted_zero,
                            _copy_x_i[:, cum_insertion + j :],
                        ),
                        dim=1,
                    ).to(self.device)
                    load = 1
                    cum_insertion += 1
                else:
                    load -= self.depot_node_demand[0, j]
            if _copy_x_i[0, -1] != 0:
                _copy_x_i = torch.cat(
                    (_copy_x_i, torch.zeros((1, 1), dtype=torch.int64).to(self.device)),
                    dim=1,
                )
            repaired_x.append(_copy_x_i)

        can_be_concated = False

        try:
            x_batch = torch.cat(repaired_x, dim=0).to(self.device)
            can_be_concated = True
        except:
            pass

        if can_be_concated:
            batch_size = x_batch.shape[0]
            res = (
                self.evaluate_by_batch(batch_size, x_batch)
                .reshape((x.shape[0], self.n_obj))
                .cpu()
                .numpy()
            )

        else:
            res = np.zeros((0, self.n_obj))
            for x_i in repaired_x:
                res_i = (
                    self.evaluate_by_batch(batch_size=1, x_batch=x_i)
                    .reshape((1, self.n_obj))
                    .cpu()
                    .numpy()
                )
                res = np.concatenate((res, res_i), axis=0)

        out["F"] = res

    def evaluate_by_batch(self, batch_size, x_batch):
        self.batch_size = batch_size
        self.selected_node_list = x_batch.reshape((batch_size, 1, -1))
        if self.depot_node_xy.shape[0] == 1:
            self.depot_node_xy = self.depot_node_xy.repeat(batch_size, 1, 1)
        if self.depot_node_demand.shape[0] == 1:
            self.depot_node_demand = self.depot_node_demand.repeat(batch_size, 1)
        res = self._get_travel_distance()
        return res

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        # obj1: travel_distances
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)

        # obj2: makespans
        not_idx = (gathering_index[:, :, :, 0] > 0).roll(dims=2, shifts=-1)
        cum_lengths = torch.cumsum(segment_lengths, dim=2)

        # cum_lengths[not_idx] = 0
        cum_lengths[not_idx] = 0
        sorted_cum_lengths, _ = cum_lengths.sort(axis=2)

        rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts=1)
        diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
        diff_mat[diff_mat < 0] = 0

        makespans, _ = torch.max(diff_mat, dim=2)

        objs = torch.stack([travel_distances, makespans], axis=2)

        return objs


class BiCVRP100(MOCVRP):
    def __init__(self):
        super().__init__(problem_size=100)

    def get_nadir_point(self):
        return np.array([46.15801239, 8.90400982])

    def get_ideal_point(self):
        return np.array([15.86342144, 2.16334248])


class BiCVRP50(MOCVRP):
    def __init__(self):
        super().__init__(problem_size=50)

    def get_nadir_point(self):
        return np.array([34.34570312, 8.17745972])

    def get_ideal_point(self):
        return np.array([9.49677372, 1.9455843])


class BiCVRP20(MOCVRP):
    def __init__(self):
        super().__init__(problem_size=20)

    def get_nadir_point(self):
        return np.array([12.5288353, 5.61969852])

    def get_ideal_point(self):
        return np.array([5.33629513, 1.77503395])
