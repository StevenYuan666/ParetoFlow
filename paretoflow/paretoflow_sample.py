"""
This file contains the implementation of the ParetoFlowSampler class,
which is used to generate samples
"""

import math
from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from pymoo.util.ref_dirs import get_reference_directions
from tqdm import tqdm

from paretoflow.flow_net import FlowMatching
from paretoflow.multiple_model_predictor_net import MultipleModels
from paretoflow.paretoflow_utils import get_N_non_dominated_solutions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParetoFlowSampler:
    """
    This class contains the implementation of the ParetoFlowSampler class,
    which is used to generate samples.
    """

    def __init__(
        self,
        fm_model: FlowMatching,
        proxies: MultipleModels,
    ):
        """
        Initialize the ParetoFlowSampler class.
        :param fm_model: FlowMatching: the FlowMatching model
        :param proxies: MultipleModels: the MultipleModels model
        """
        self.fm_model = fm_model.to(device)
        self.proxies = proxies.to(device)
        self.device = device
        self.D = fm_model.D
        self.sigma = fm_model.sigma
        self.vnet = fm_model.vnet
        self.time_embedding = fm_model.time_embedding
        self.classifiers = [
            proxy.to(self.device) for proxy in proxies.obj2model.values()
        ]

    def weighted_conditional_vnet(
        self,
        x_t: torch.Tensor,
        t: float,
        weights: torch.Tensor,
        gamma: float = 2.0,
        t_threshold: float = 0.8,
        **kwargs
    ) -> torch.Tensor:
        """
        x_t: the noise sample at time t
        t: the time
        classifiers: a list of classifiers, each is a function that
        takes x and returns the score on the i-th objective
        weights: the weights for each objective
        return: the conditional vector field at time t

        Since we know u_t(x|x_1) = (x_1 - (1 - sigma) * x) / (1 - (1 - sigma) * t)
        We can solve for x_1

        For the classifiers guidance, since we are using the proxy models,
        they are not outputting the probabilities but the scores.
        However, we can treat them as when conditioning on score of 1,
        the predicted scores is the probability of the sample has the score of 1.
        Therefore, we can use the log-likelihood to guide the sampling process
        """
        x_t_ = x_t.detach().requires_grad_(True)
        # to calculate the gradients of x_t, start a new computation graph,
        # shape: (batch_size, D)
        with torch.no_grad():
            time_embedding = self.time_embedding(
                torch.Tensor([t]).to(x_t_.device)
            )  # shape: (batch_size, D)
            x_t_with_time_embedding = (
                x_t_.detach().data + time_embedding
            )  # shape: (batch_size, D)
            # Since the classifiers are trained for x_1, we need to convert the samples
            # to x_1 first
            u_t = self.vnet(x_t_with_time_embedding)  # shape: (batch_size, D)
            if t < t_threshold:
                return u_t
        x_1 = (
            u_t * (1 - (1 - self.sigma) * t) + (1 - self.sigma) * x_t_
        )  # to use eq6 in the paper, x_1 is a function of x_t_, shape: (batch_size, D)

        # calculate the scores from the classifiers as eq 9 in the paper, don't need log here
        scores = []  # shape: (batch_size, len(classifiers))
        for classifier in self.classifiers:
            scores.append(-1 * classifier(x_1))
        # shape: (batch_size, len(classifiers))
        scores = torch.stack(scores, dim=1).squeeze()
        log_value = scores * weights  # shape: (batch_size, len(classifiers))
        log_value = torch.sum(log_value, dim=1)  # shape: (batch_size)
        log_value = torch.sum(
            log_value
        )  # need a summation again since backward only supported for scalar value, shape: scalar
        log_value.backward()

        return (
            u_t + gamma * (1 - t) / max(t, kwargs["delta_t"]) * x_t_.grad
        )  # align eq5 in the paper, shape: (batch_size, D)

    @classmethod
    def get_neighborhood_indices(
        cls,
        weight: torch.Tensor,
        objectives_weights: torch.Tensor,
        K: int,
        distance: str = "cosine",
    ) -> torch.Tensor:
        """
        weight: the weight for the objective
        objectives_weights: the weights for each objective
        K: the number of samples we want to include in the neighborhood
        distance: the distance metric to calculate the distance
        between the weight and the weights of the objectives. The default is cosine angle
        distance, but can be changed to Euclidean distance
        return: the indices of the neighborhood samples
        """
        if distance == "cosine":
            # calculate the cosine similarity between the weight and the weights of
            # the objectives
            cos = nn.CosineSimilarity(dim=1)
            cos_similarities = cos(objectives_weights, weight.unsqueeze(0))
            # sort the cosine similarities and get the indices of the K nearest neighbors
            _, indices = torch.topk(cos_similarities, K, largest=True)
            return indices

        if distance == "euclidean":
            # calculate the distance between the weight and the weights of
            # the objectives based on the Euclidean distance
            distances = torch.norm(objectives_weights - weight, dim=1)
            # sort the distances and get the indices of the K nearest neighbors
            _, indices = torch.topk(distances, K, largest=False)
            return indices

    # Get Objectives Weights and Number of Samples we want to generate
    @classmethod
    def calculate_objectives_weights(
        cls, M: int, num_solutions: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Get weights for each objective, number of objectives is equal to number of classifiers
        n_partitions is equal to the number of samples we want to generate,
        because we want to generate a batch of
        samples to maximize different objectives, so that can maximize the hypervolume
        len(objectives_weights) = combination(M + n_partitions - 1, n_partitions)
        where M is the number of objectives

        M: the number of objectives
        num_solutions: the number of new solutions we want to keep in the final pareto set
        the number of samples needed to generate should be larger than or equal to num_solutions
        return: objective weights, the number of samples we want to generate
        """
        # len(objectives_weights) = combination(M + n_partitions - 1, n_partitions)
        # where M is the number of objectives
        # We hope the len(objectives_weights) is equal to batch_size or larger than batch_size
        n_partitions = 1
        while True:
            if math.comb(M + n_partitions - 1, n_partitions) >= 400:
                break
            n_partitions += 1
        objectives_weights = get_reference_directions(
            "uniform", M, n_partitions=n_partitions
        )
        batch_size = objectives_weights.shape[0]

        # shape: (batch_size, len(classifiers))
        objectives_weights = torch.tensor(objectives_weights).to(device)

        return objectives_weights, batch_size

    @classmethod
    def calculate_angles(cls, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: the first tensor, shape: (n, D)
        b: the second tensor, shape: (n, D)
        return: the angle between the two tensors, shape: (n, 1)
        """
        inner_product = (a * b).sum(dim=1)
        a_norm = a.pow(2).sum(dim=1).pow(0.5)
        b_norm = b.pow(2).sum(dim=1).pow(0.5)
        cos = inner_product / (a_norm * b_norm)
        angle = torch.acos(cos)
        return angle.unsqueeze(1)

    @classmethod
    def get_ts_and_delta_t(
        cls, T: int, t_threshold: float = 0.8, adaptive: bool = False
    ) -> Tuple[torch.Tensor, Callable]:
        """
        T: the number of steps to generate the samples
        t_threshold: the threshold to switch the sampling method
        adaptive: whether to use the adaptive time steps
        return: the time steps and the step size
        """
        if adaptive:
            ts1 = torch.linspace(0.0, t_threshold, int(T * (1 - t_threshold)))
            ts2 = torch.linspace(t_threshold, 1.0, T - int(T * (1 - t_threshold)))
            ts = torch.cat((ts1, ts2))
            assert ts.shape[0] == T
            delta_t1 = (
                ts1[1] - ts1[0] if ts1.shape[0] > 1 else 1.0
            )  # Handle the case when t_threshold = 1
            delta_t2 = (
                ts2[1] - ts2[0] if ts2.shape[0] > 1 else delta_t1
            )  # Handle the case when t_threshold = 0

            def delta_t(t):
                if t < t_threshold:
                    return delta_t1
                return delta_t2

            return ts, delta_t
        else:
            ts = torch.linspace(0.0, 1.0, T)
            d_t = ts[1] - ts[0]

            def delta_t(t):
                return d_t

            return ts, delta_t

    def initialize_pareto_set(
        self,
        batch_size: int,
        objectives_weights: torch.Tensor = None,
        methods: str = "d_best",
        all_x: np.ndarray = None,
        all_y: np.ndarray = None,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        batch_size: the number of samples we want to generate
        objectives_weights: the weights for each objective
        methods: the method to initialize the pareto set
        all_x: the existing solutions
        all_y: the existing scores
        return: the pareto set of the generated samples
        """

        # Initialize the pareto set with empty values
        if methods == "empty_init":
            pareto_set = [(torch.empty(self.D), float("-inf"))] * batch_size
            return pareto_set
        # Initialize the pareto set with the existing best samples from the offline dataset
        elif methods == "d_best":
            assert (
                objectives_weights is not None
            ), "Error: The objectives_weights_list should be provided"
            assert (
                len(objectives_weights) == batch_size
            ), "Error: Length of objectives_weights_list must equal batch_size"

            # Get all solutions from the offline dataset
            all_x, all_y = get_N_non_dominated_solutions(
                N=batch_size, all_x=all_x, all_y=all_y
            )

            # Convert to tensors
            all_x = torch.tensor(all_x).to(device)
            all_y = torch.tensor(all_y).to(device)

            # Initialize Pareto set
            pareto_set = []
            remaining_indices = torch.arange(all_x.size(0)).to(device)

            for i in range(batch_size):
                weights = objectives_weights[i].to(
                    device
                )  # Weight vector for position i

                # Compute scalarized scores
                scalarized_scores = torch.matmul(all_y, weights)

                # Select the candidate with the best scalarized score
                best_index = torch.argmax(scalarized_scores)
                best_x = all_x[best_index]
                best_score = scalarized_scores[best_index]

                # Add to Pareto set
                pareto_set.append((best_x, best_score))

                # Remove the selected candidate from consideration
                mask = torch.ones(all_x.size(0), dtype=torch.bool).to(device)
                mask[best_index] = False
                all_x = all_x[mask]
                all_y = all_y[mask]
                remaining_indices = remaining_indices[mask]

            return pareto_set
        else:
            raise ValueError("Invalid method for initializing the pareto set")

    @classmethod
    def all_neighborhood_indices(
        cls,
        batch_size: int,
        objectives_weights: torch.Tensor,
        K: int,
        distance: str = "cosine",
    ) -> torch.Tensor:
        """
        batch_size: the number of samples we want to generate
        objectives_weights: the weights for each objective
        K: the number of samples we want to include in the neighborhood
        distance: the distance metric to calculate the distance
        between the weight and the weights of the objectives. The default is cosine angle
        distance, but can be changed to Euclidean distance
        return: the indices of the neighborhood samples
        """
        # Calculate the neighborhood of the diverse samples
        neighborhood_indices = []
        for i in range(batch_size):
            neighborhood_indices.append(
                cls.get_neighborhood_indices(
                    objectives_weights[i],
                    objectives_weights,
                    K,
                    distance=distance,
                )
            )
        # shape: (batch_size, K)
        neighborhood_indices = torch.stack(neighborhood_indices, dim=0)
        return neighborhood_indices

    def calculate_scores(
        self,
        batch_size: int,
        t: float,
        O: int,
        batch_diverse_samples: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch_size: the number of samples we want to generate
        t: the time
        O: the number of offspring samples we want to generate
        batch_diverse_samples: the diverse samples
        return: the scores for the samples, merged_samples_x_1, shape: (batch_size * O, D)
        """
        # Predict the scores for the diverse samples
        scores = []  # shape: (batch_size, O, len(classifiers))
        # shape: (batch_size * O, D)
        merged_samples = batch_diverse_samples.view(-1, self.D)
        with torch.no_grad():
            # Since the classifiers are trained for x_1, we need to convert the samples
            # to x_1 first
            # shape: (batch_size * O, D)
            time_embedding = self.time_embedding(
                torch.Tensor([t]).to(merged_samples.device)
            ).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            merged_samples_with_time_embedding = merged_samples + time_embedding
            # shape: (batch_size * O, D)
            merged_samples_u_t = self.vnet(merged_samples_with_time_embedding)
            merged_samples_x_1 = (
                merged_samples_u_t * (1 - (1 - self.sigma) * t)
                + (1 - self.sigma) * merged_samples
            )
            if "need_repair" in kwargs and kwargs["need_repair"]:
                xl = kwargs["xl"]
                xu = kwargs["xu"]
                merged_samples_x_1 = torch.clip(merged_samples_x_1, xl, xu)
            for classifier in self.classifiers:
                scores.append(-1 * classifier(merged_samples_x_1))

        # shape: (batch_size * O, len(classifiers))
        scores = torch.stack(scores, dim=1)
        # shape: (batch_size, O, len(classifiers))
        scores = scores.view(batch_size, O, len(self.classifiers))
        # shape: (batch_size, O, len(classifiers))
        merged_samples_x_1 = merged_samples_x_1.view(batch_size, O, self.D)

        return scores, merged_samples_x_1

    @classmethod
    def get_angle_filter_mask(
        cls, angles: torch.Tensor, phi: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        angles: the angles between the predicted scores and the i-th objective weights
        phi: the threshold for the angles
        batch_size: the number of samples we want to generate
        return: the filter mask for the angles
        """
        # shape: (batch_size, K * O, 1)
        angle_filter_mask = angles <= (1 / 2 * phi.unsqueeze(1))
        # # Set mask to all False
        # angle_filter_mask[:] = False
        # If there is no sample that satisfies the condition,
        # we keep the sample with the smallest angle
        # shape: (batch_size, 1)
        for i in range(batch_size):
            angle_filter_mask[i, torch.argmin(angles[i])] = True
        # If the angle is smaller than phi_i, we keep the sample, otherwise we set the score to -inf
        return angle_filter_mask

    def repair_boundary(
        self, solutions: torch.Tensor, t: float, xl: torch.Tensor, xu: torch.Tensor
    ) -> torch.Tensor:
        """
        solutions: the solutions to be repaired  # shape: (batch_size * O, D)
        xl: the lower bound of the solutions
        xu: the upper bound of the solutions
        t: the time
        return: the repaired solutions # shape: (batch_size * O, D)
        """
        # Calculate the lower bound and upper bound at time t
        # shape: (batch_size * O, D), get u_t first
        with torch.no_grad():
            time_embedding = self.time_embedding(
                torch.Tensor([t]).to(solutions.device)
            ).repeat(solutions.shape[0], 1)
            solutions_with_time_embedding = solutions + time_embedding
            u_t = self.vnet(solutions_with_time_embedding)

        # Calculate xl and xu at time t
        xl_t = (xl - (1 - (1 - self.sigma) * t) * u_t) / (1 - self.sigma)
        xu_t = (xu - (1 - (1 - self.sigma) * t) * u_t) / (1 - self.sigma)

        # Repair the boundary of the solutions
        # shape: (batch_size * O, D)
        x_t = torch.clip(solutions, xl_t, xu_t)

        return x_t

    @classmethod
    def check_duplicates(cls, pareto_set: List[Tuple[torch.Tensor, float]]):
        """
        pareto_set: the pareto set to be checked. Tuple of (solution, score),
        solution is a tensor, score is a float
        return: the number of duplicates in the pareto set
        """
        # Get all solutions, change dtype to float tensor
        solutions = [solution[0].float() for solution in pareto_set]
        # Check duplicates, if the two tensors are close enough, we consider them as duplicates
        unique_solutions = []
        for solution in solutions:
            if unique_solutions == []:
                unique_solutions.append(solution)
                continue
            else:
                is_duplicate = False
                for unique_solution in unique_solutions:
                    if torch.allclose(solution, unique_solution, atol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_solutions.append(solution)
        return len(solutions) - len(unique_solutions)

    @classmethod
    def remove_duplicates(cls, pareto_set: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        pareto_set: the pareto set to be checked. List of solutions, each is a tensor
        return: the pareto set without duplicates
        """
        pareto_set = [solution.float() for solution in pareto_set]
        unique_solutions = []
        for solution in pareto_set:
            if unique_solutions == []:
                unique_solutions.append(solution)
                continue
            else:
                is_duplicate = False
                for unique_solution in unique_solutions:
                    if torch.allclose(solution, unique_solution, atol=1e-3):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_solutions.append(solution)
        return unique_solutions

    def sample(
        self,
        all_x: np.ndarray,
        all_y: np.ndarray,
        T: int = 1000,
        O: int = 5,
        K: int = 0,
        num_solutions: int = 256,
        distance: str = "cosine",
        g_t: float = 0.1,
        init_method: str = "d_best",
        t_threshold: float = 0.8,
        adaptive: bool = False,
        gamma: float = 2.0,
        xl: np.ndarray = None,
        xu: np.ndarray = None,
    ) -> List[torch.Tensor]:
        """
        all_x: the existing solutions
        all_y: the existing scores
        T: the number of steps to generate the samples
        O: the number of offspring samples we want to generate
        K: the number of samples we want to include in the neighborhood
        num_solutions: the number of new solutions we want to keep in the final pareto set
        distance: the distance metric to calculate the distance
        between the weight and the weights of the objectives. The default is cosine angle
        distance, but can be changed to Euclidean distance
        g_t: the noise level
        init_method: the method to initialize the pareto set
        t_threshold: the threshold to switch the sampling method
        adaptive: whether to use the adaptive time steps
        gamma: the gamma parameter
        xl: the lower bound of the solutions
        xu: the upper bound of the solutions
        return: the pareto set of the generated samples
        """
        if K == 0:
            K = len(self.classifiers) + 1
        # Obtain objectives weights and the number of samples we want to generate
        # shape: (batch_size, len(classifiers))
        objectives_weights, batch_size = ParetoFlowSampler.calculate_objectives_weights(
            len(self.classifiers), num_solutions
        )

        # the pareto set of the generated samples
        # shape: (batch_size, D)
        pareto_set = self.initialize_pareto_set(
            batch_size,
            objectives_weights=objectives_weights,
            methods=init_method,
            all_x=all_x,
            all_y=all_y,
        )

        # Calculate the neighborhood of the diverse samples
        # shape: (batch_size, K)
        neighborhood_indices = ParetoFlowSampler.all_neighborhood_indices(
            batch_size, objectives_weights, K, distance=distance
        )

        # Calculate the M closest indices for filtering the samples in non-convex cases
        # shape: (batch_size, M)
        M = len(self.classifiers) + 1
        all_m_closest_indices = ParetoFlowSampler.all_neighborhood_indices(
            batch_size, objectives_weights, M, distance=distance
        )

        # shape: (batch_size, M, len(classifiers))
        m_closest_objectives_weights = objectives_weights[all_m_closest_indices]
        # shapeL (batch_size * M, 1)
        m_closest_angles = ParetoFlowSampler.calculate_angles(
            m_closest_objectives_weights.view(batch_size * M, len(self.classifiers)),
            objectives_weights.repeat_interleave(
                M, dim=0
            ),  # shape: (batch_size, 1, len(classifiers))
        )
        m_closest_angles = m_closest_angles.view(batch_size, M, 1)
        # shape: (batch_size, 1)
        # #We find the M closest angles but include the angle with itself,
        # so we divide by len(classifiers) to get the average angle
        # because the angle with itself is 0
        phi = 2 * m_closest_angles.sum(dim=1) / len(self.classifiers)

        # Algorithm 1 in the paper
        # go step-by-step to x_1 (data)
        ts, delta_t = ParetoFlowSampler.get_ts_and_delta_t(
            T, t_threshold=t_threshold, adaptive=adaptive
        )

        # Precompute lower bound and upper bound for the repair method
        if xl is not None and xu is not None:
            # shape: (D)
            xl = torch.from_numpy(
                xl
            ).squeeze()  # Add squeeze to remove the first dimension of size 1
            # shape: (batch_size * O, D)
            xl = xl.unsqueeze(0).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            xl = xl.to(device).type(torch.float32)
            # shape: (D)
            xu = torch.from_numpy(
                xu
            ).squeeze()  # Add squeeze to remove the first dimension of size 1
            # shape: (batch_size * O, D)
            xu = xu.unsqueeze(0).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            xu = xu.to(device).type(torch.float32)
            need_repair = True
        else:
            xl = None
            xu = None
            need_repair = False

        # use tqdm for progress bar
        with tqdm(total=T, desc="Conditional Sampling", unit="step") as pbar:
            # sample x_0 first, offspring
            # shape: (batch_size, D)
            x_t = self.fm_model.sample_base(torch.empty(batch_size, self.D)).to(device)

            # Euler method
            for t in ts[1:]:
                # this is x_t + v(x_t, t, y) * delta_t
                # shape: (batch_size, D)
                x_t = x_t + self.weighted_conditional_vnet(
                    x_t,
                    t - delta_t(t),
                    weights=objectives_weights,
                    gamma=gamma,
                    t_threshold=t_threshold,
                    delta_t=delta_t(t),
                ) * delta_t(t)
                if t < t_threshold:
                    if need_repair:
                        x_t = self.repair_boundary(
                            x_t,
                            t,
                            xl[0].unsqueeze(0).repeat(batch_size, 1),
                            xu[0].unsqueeze(0).repeat(batch_size, 1),
                        )
                    pbar.update(1)
                    continue

                # Stochastic Euler method to generate diverse samples
                # shape: (batch_size, O, D)
                batch_diverse_samples = x_t.unsqueeze(1).repeat(1, O, 1)
                # shape: (batch_size, O, D)
                batch_diverse_samples = batch_diverse_samples + g_t * torch.randn_like(
                    batch_diverse_samples
                ) * torch.sqrt(delta_t(t))

                # Repair the boundary of the samples
                if need_repair:
                    # x_t = self.repair_boundary(x_t, t, xl, xu)
                    batch_diverse_samples = batch_diverse_samples.view(
                        batch_size * O, self.D
                    )
                    batch_diverse_samples = self.repair_boundary(
                        batch_diverse_samples, t, xl, xu
                    )
                    batch_diverse_samples = batch_diverse_samples.view(
                        batch_size, O, self.D
                    )

                    # Calculate the scores for the diverse samples
                    # shape: (batch_size, O, len(classifiers))
                    scores, merged_samples_x_1 = self.calculate_scores(
                        batch_size,
                        t,
                        O,
                        batch_diverse_samples,
                        need_repair=True,
                        xl=xl,
                        xu=xu,
                    )
                else:
                    scores, merged_samples_x_1 = self.calculate_scores(
                        batch_size, t, O, batch_diverse_samples
                    )

                # Calculate the scores for the neighborhood samples
                # Filter the samples to avoid non-convexity as eq 11 in the paper

                # shape: (batch_size, K, O, len(classifiers))
                neighborhood_scores = scores[neighborhood_indices]
                # shape: (batch_size, K * O, len(classifiers))
                neighborhood_scores = neighborhood_scores.view(
                    batch_size, K * O, len(self.classifiers)
                )

                # Calculate the angles between the predicted scores and the i-th objective weights
                # shape: (batch_size, K * O, 1)
                angles = ParetoFlowSampler.calculate_angles(
                    neighborhood_scores.view(batch_size * K * O, len(self.classifiers)),
                    objectives_weights.repeat_interleave(K * O, dim=0),
                )
                angles = angles.view(batch_size, K * O, 1)

                # Filter out the samples whose angles are larger than phi_i, find the indices
                # shape: (batch_size, K * O, 1)
                angle_filter_mask = ParetoFlowSampler.get_angle_filter_mask(
                    angles, phi, batch_size
                )

                # Calculate weighted sum of the scores using the i-th objective weights
                # shape: (batch_size, K * O, len(classifiers))
                weighted_scores = neighborhood_scores * objectives_weights.unsqueeze(
                    1
                ).repeat_interleave(K * O, dim=1)
                # shape: (batch_size, K * O, 1)
                weighted_scores = weighted_scores.sum(dim=-1).unsqueeze(-1)
                # shape (batch_size, K * O, 1)
                weighted_scores[~angle_filter_mask] = float("-inf")
                # Choose the sample with the highest score as the next offspring
                # shape: (batch_size)
                index = torch.argmax(weighted_scores.squeeze(), dim=1)

                # shape: (batch_size, K, O, D)
                neighborhood_designs = batch_diverse_samples[neighborhood_indices]
                # shape: (batch_size, K * O, D)
                neighborhood_designs = neighborhood_designs.view(
                    batch_size, K * O, self.D
                )
                # Use the index to get the next offspring
                # shape: (batch_size, D)
                next_offspring = neighborhood_designs[torch.arange(batch_size), index]

                # Update the pareto set. If the new offspring is better
                # than the i-th solution in the pareto set,
                # replace the i-th solution with the new offspring
                for i in range(batch_size):
                    if weighted_scores[i, index[i]] > pareto_set[i][1]:
                        # shape: (K, O, D)
                        candidates = merged_samples_x_1[neighborhood_indices[i]]
                        # shape: (K * O, D)
                        candidates = candidates.view(K * O, self.D)
                        # shape: (D)
                        pareto_set[i] = (
                            candidates[index[i]].squeeze(),
                            weighted_scores[i, index[i]],
                        )
                # Update x_t
                x_t = next_offspring
                pbar.update(1)

        temp_pareto_set = [pareto_set[i][0] for i in range(batch_size)]
        # Remove duplicates in the pareto set, because they are not contributing to the hypervolume
        temp_pareto_set = ParetoFlowSampler.remove_duplicates(temp_pareto_set)
        assert (
            len(temp_pareto_set) >= num_solutions
        ), "Error: The number of solutions in the pareto set is \
        less than the number of solutions we want to keep"
        res_x = torch.stack(temp_pareto_set, dim=0).squeeze()
        res_y = []
        with torch.no_grad():
            for classifier in self.classifiers:
                res_y.append(-1 * classifier(res_x.to(device)))
        res_y = torch.stack(res_y, dim=1).squeeze()
        res_x = res_x.cpu().detach().numpy()
        res_y = res_y.cpu().detach().numpy()
        # Do a non-dominated sorting to get the pareto set
        res_x, res_y = get_N_non_dominated_solutions(num_solutions, res_x, res_y)
        visible_masks = np.ones(len(res_y))
        visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
        visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
        res_x = res_x[np.where(visible_masks == 1)[0]]
        res_y = res_y[np.where(visible_masks == 1)[0]]
        return res_x, res_y
