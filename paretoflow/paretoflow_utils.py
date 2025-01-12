"""
This file contains utility functions for the ParetoFlow package.
"""

from typing import List

import numpy as np
import torch
from pymoo.util.misc import find_duplicates
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def get_N_non_dominated_solutions(N: int, all_x: np.ndarray, all_y: np.ndarray):
    """
    Get N non-dominated solutions from the input data.
    :param N: int: the number of non-dominated solutions
    :param all_x: np.ndarray: the input data
    :param all_y: np.ndarray: the output data
    :return: np.ndarray: the non-dominated solutions
    """
    N_best_indexes = get_N_nondominated_indices(
        Y=all_y, num_ret=N, fronts=NonDominatedSorting().do(all_y)
    )

    return (
        all_x[N_best_indexes],
        all_y[N_best_indexes],
    )


tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def calc_crowding_distance(F) -> np.ndarray:
    """
    Calculate the crowding distance for the objective space values.
    :param F: np.ndarray: the objective space values
    :return: np.ndarray: the crowding distance
    """

    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float("inf"), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full(
        (1, n_obj), float("-inf"), device=F.device, dtype=F.dtype
    )
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat(
        [neg_inf_tensor, F_sorted], dim=0
    )

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float("nan")

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    # if we divide by zero because all values in one column are equal replace by none
    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = (
        torch.sum(
            torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J), dim=1
        )
        / n_obj
    )

    return crowding_dist.detach().cpu().numpy()


def get_N_nondominated_indices(Y, num_ret, fronts=None) -> List[int]:
    """
    Get the indices of the N non-dominated solutions.
    :param Y: np.ndarray: the objective space values
    :param num_ret: int: the number of non-dominated solutions
    :param fronts: List: the fronts
    :return: List[int]: the indices of the non-dominated solutions
    """
    assert num_ret >= 0
    if num_ret == 0:
        return []
    if fronts is None:
        fronts = NonDominatedSorting().do(Y, return_rank=True, n_stop_if_ranked=num_ret)
    indices_cnt = 0
    indices_select = []
    for front in fronts:
        if indices_cnt + len(front) < num_ret:
            indices_cnt += len(front)
            indices_select += [int(i) for i in front]
        else:
            n_keep = num_ret - indices_cnt
            F = Y[front]

            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]

            _F = F[is_unique]
            _d = calc_crowding_distance(_F)

            d = np.zeros(len(front))
            d[is_unique] = _d
            I = np.argsort(d)[-n_keep:]
            indices_select += [int(i) for i in I]
            break

    return indices_select
