"""This module contains the neural network for the GFMO-Guided Flows in
Multi-Objective Optimization."""

import math

import numpy as np
import torch
import torch.nn as nn
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm

from gfmo_utils import get_reference_directions
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.off_moo_bench.problem.dtlz import DTLZ
from offline_moo.off_moo_bench.problem.synthetic_func import SyntheticProblem
from offline_moo.off_moo_bench.utils import get_N_nondominated_indices
from offline_moo.utils import get_quantile_solutions

# get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a simple neural network
class VectorFieldNet(nn.Module):
    def __init__(self, D, M=512):
        super(VectorFieldNet, self).__init__()

        self.D = D
        self.M = M
        self.net = nn.Sequential(
            nn.Linear(D, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, M),
            nn.SELU(),
            nn.Linear(M, D),
        )

    def forward(self, x):
        return self.net(x)


class FlowMatching(nn.Module):
    def __init__(self, vnet, sigma, D, T, stochastic_euler=False, prob_path="icfm"):
        super(FlowMatching, self).__init__()

        self.vnet = vnet

        self.time_embedding = nn.Sequential(nn.Linear(1, D))

        # other params
        self.D = D

        self.T = T

        self.sigma = sigma

        self.stochastic_euler = stochastic_euler

        assert prob_path in [
            "icfm",
            "fm",
        ], (
            f"Error: The probability path could be either Independent CFM (icfm) "
            f"or Lipman's Flow Matching (fm) but {prob_path} was provided."
        )
        self.prob_path = prob_path

        self.PI = torch.from_numpy(np.asarray(np.pi))

    def log_p_base(self, x, reduction="sum", dim=1):
        log_p = -0.5 * torch.log(2.0 * self.PI) - 0.5 * x**2.0
        if reduction == "mean":
            return torch.mean(log_p, dim)
        elif reduction == "sum":
            return torch.sum(log_p, dim)
        else:
            return log_p

    def sample_base(self, x_1):
        # Gaussian base distribution
        if self.prob_path == "icfm":
            return torch.randn_like(x_1)
        elif self.prob_path == "fm":
            return torch.randn_like(x_1)
        else:
            return None

    def sample_p_t(self, x_0, x_1, t):
        if self.prob_path == "icfm":
            mu_t = (1.0 - t) * x_0 + t * x_1
            sigma_t = self.sigma
        elif self.prob_path == "fm":
            mu_t = t * x_1
            sigma_t = t * self.sigma - t + 1.0

        x = mu_t + sigma_t * torch.randn_like(x_1)

        return x

    def conditional_vector_field(self, x, x_0, x_1, t):
        if self.prob_path == "icfm":
            u_t = x_1 - x_0
        elif self.prob_path == "fm":
            u_t = (x_1 - (1.0 - self.sigma) * x) / (1.0 - (1.0 - self.sigma) * t)

        return u_t

    def forward(self, x_1, reduction="mean"):
        # =====Flow Matching
        # =====
        # z ~ q(z), e.g., q(z) = q(x_0) q(x_1), q(x_0) = base, q(x_1) = empirical
        # t ~ Uniform(0, 1)
        x_0 = self.sample_base(
            x_1
        )  # sample from the base distribution (e.g., Normal(0,I))
        t = torch.rand(size=(x_1.shape[0], 1)).to(x_1.device)

        # =====
        # sample from p(x|z)
        x = self.sample_p_t(x_0, x_1, t)  # sample independent rv

        # =====
        # invert interpolation, i.e., calculate vector field v(x,t)
        t_embd = self.time_embedding(t)
        v = self.vnet(x + t_embd)

        # =====
        # conditional vector field
        u_t = self.conditional_vector_field(x, x_0, x_1, t)

        # =====LOSS: Flow Matching
        FM_loss = torch.pow(v - u_t, 2).mean(-1)

        # Final LOSS
        if reduction == "sum":
            loss = FM_loss.sum()
        else:
            loss = FM_loss.mean()

        return loss

    # This is an unconditional sampling process
    def sample(self, batch_size=64):
        # Euler method
        # sample x_0 first
        x_t = self.sample_base(torch.empty(batch_size, self.D))

        # then go step-by-step to x_1 (data)
        ts = torch.linspace(0.0, 1.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts[1:]:
            t_embedding = self.time_embedding(torch.Tensor([t]))
            x_t = x_t + self.vnet(x_t + t_embedding) * delta_t
            # Stochastic Euler method
            if self.stochastic_euler:
                x_t = x_t + torch.randn_like(x_t) * delta_t

        x_final = x_t
        return x_final

    def weighted_conditional_vnet(
        self, x_t, t, classifiers, weights, gamma=2.0, t_threshold=0.8, **kwargs
    ):
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
        x_t_ = x_t.detach().requires_grad_(
            True
        )  # to calculate the gradients of x_t, start a new computation graph, shape: (batch_size, D)
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
        for classifier in classifiers:
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
    def get_neighborhood_indices(cls, weight, objectives_weights, K, distance="cosine"):
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
    def calculate_objectives_weights(cls, M, num_solutions):
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
            if math.comb(M + n_partitions - 1, n_partitions) >= 300:
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
    def calculate_angles(cls, a, b):
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
    def get_ts_and_delta_t(cls, T, t_threshold=0.8, adaptive=False):
        """
        T: the number of steps to generate the samples
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
        self, batch_size, objectives_weights=None, task=None, methods="d_best"
    ):
        """
        batch_size: the number of samples we want to generate
        return: the pareto set of the generated samples
        """

        # Initialize the pareto set with empty values
        if methods == "empty_init":
            pareto_set = [(torch.empty(self.D), float("-inf"))] * batch_size
            return pareto_set
        # Initialize the pareto set with the existing best samples from the offline dataset
        elif methods == "d_best":
            assert task is not None, "Error: The task should be provided"
            assert (
                objectives_weights is not None
            ), "Error: The objectives_weights_list should be provided"
            assert (
                len(objectives_weights) == batch_size
            ), "Error: Length of objectives_weights_list must equal batch_size"

            # Get all solutions from the offline dataset
            all_x, all_y = task.get_N_non_dominated_solutions(
                N=batch_size, return_x=True, return_y=True
            )

            # Preprocess inputs
            if task.is_discrete:
                all_x = task.to_logits(all_x)
                _, dim, n_classes = all_x.shape
                all_x = all_x.reshape(-1, dim * n_classes)
            if task.is_sequence:
                all_x = task.to_logits(all_x)
            all_x = task.normalize_x(all_x)
            all_y = task.normalize_y(all_y)

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
        cls, batch_size, objectives_weights, K, distance="cosine"
    ):
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
        self, batch_size, t, O, classifiers, batch_diverse_samples, **kwargs
    ):
        """
        classifiers: a list of classifiers, each is a function that takes x
        and returns the score on the i-th objective
        merged_samples_x_1: the samples to predict the scores
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
            for classifier in classifiers:
                scores.append(-1 * classifier(merged_samples_x_1))

        # shape: (batch_size * O, len(classifiers))
        scores = torch.stack(scores, dim=1)
        # shape: (batch_size, O, len(classifiers))
        scores = scores.view(batch_size, O, len(classifiers))
        # shape: (batch_size, O, len(classifiers))
        merged_samples_x_1 = merged_samples_x_1.view(batch_size, O, self.D)

        return scores, merged_samples_x_1

    @classmethod
    def get_angle_filter_mask(cls, angles, phi, batch_size):
        """
        angles: the angles between the predicted scores and the i-th objective weights
        phi: the threshold for the angles
        return: the filter mask for the angles
        """
        # shape: (batch_size, K * O, 1)
        angle_filter_mask = angles <= (1 / 2 * phi.unsqueeze(1))
        # # Set mask to all False
        # angle_filter_mask[:] = False
        # If there is no sample that satisfies the condition, we keep the sample with the smallest angle
        # shape: (batch_size, 1)
        for i in range(batch_size):
            angle_filter_mask[i, torch.argmin(angles[i])] = True
        # If the angle is smaller than phi_i, we keep the sample, otherwise we set the score to -inf
        return angle_filter_mask

    def repair_boundary(self, solutions, t, xl, xu):
        """
        solutions: the solutions to be repaired  # shape: (batch_size * O, D)
        task: the task to be solved
        xl: the lower bound of the solutions
        xu: the upper bound of the solutions
        convert_to_x1: whether to convert the solutions to x_1 first
        method: the method to repair the solutions
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
    def check_duplicates(cls, pareto_set):
        """
        pareto_set: the pareto set to be checked. Tuple of (solution, score), solution is a tensor, score is a float
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
    def remove_duplicates(cls, pareto_set):
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

    @classmethod
    def get_N_non_dominated_solutions(cls, res_x, res_y, N, classifiers):
        predicted_scores = []
        with torch.no_grad():
            for classifier in classifiers:
                predicted_scores.append(-1 * classifier(res_x))
        predicted_res_y = torch.stack(predicted_scores, dim=1).squeeze()
        fronts = NonDominatedSorting().do(predicted_res_y)
        N_best_indices = get_N_nondominated_indices(
            Y=predicted_res_y, num_ret=N, fronts=fronts
        )
        return res_x[N_best_indices], res_y[N_best_indices]

    def gfmo_sample(
        self,
        classifiers,
        T=1000,
        O=10,
        K=20,
        num_solutions=256,
        distance="cosine",
        g_t=0.1,
        init_method="d_best",
        task=None,
        task_name=None,
        t_threshold=0.8,
        adaptive=False,
        gamma=2.0,
    ):
        """
        classifiers: a list of classifiers, each is a function that takes x
        and returns the score on the i-th objective
        T: the number of steps to run the algorithm
        O: the number of offspring to generate at each step
        K: the number of neighborhoods to consider
        num_solutions: the number of solutions to generate
        distance: the distance metric to use
        g_t: the step size for the stochastic Euler method
        init_method: the method to initialize the pareto set
        t_threshold: the threshold for the time step
        adaptive: whether to use the adaptive time step
        gamma: the gamma parameter for the weighted sum of the scores
        """
        # Obtain objectives weights and the number of samples we want to generate
        # shape: (batch_size, len(classifiers))
        objectives_weights, batch_size = FlowMatching.calculate_objectives_weights(
            len(classifiers), num_solutions
        )

        # the pareto set of the generated samples
        # shape: (batch_size, D)
        pareto_set = self.initialize_pareto_set(
            batch_size,
            objectives_weights=objectives_weights,
            task=task,
            methods=init_method,
        )

        # Calculate the neighborhood of the diverse samples
        # shape: (batch_size, K)
        neighborhood_indices = FlowMatching.all_neighborhood_indices(
            batch_size, objectives_weights, K, distance=distance
        )

        # Calculate the M closest indices for filtering the samples in non-convex cases
        # shape: (batch_size, M)
        M = len(classifiers) + 1
        all_m_closest_indices = FlowMatching.all_neighborhood_indices(
            batch_size, objectives_weights, M, distance=distance
        )

        # shape: (batch_size, M, len(classifiers))
        m_closest_objectives_weights = objectives_weights[all_m_closest_indices]
        # shapeL (batch_size * M, 1)
        m_closest_angles = FlowMatching.calculate_angles(
            m_closest_objectives_weights.view(batch_size * M, len(classifiers)),
            objectives_weights.repeat_interleave(
                M, dim=0
            ),  # shape: (batch_size, 1, len(classifiers))
        )
        m_closest_angles = m_closest_angles.view(batch_size, M, 1)
        # shape: (batch_size, 1)
        # #We find the M closest angles but include the angle with itself, so we divide by len(classifiers) to get the average angle
        # because the angle with itself is 0
        phi = 2 * m_closest_angles.sum(dim=1) / len(classifiers)

        # Algorithm 1 in the paper
        # go step-by-step to x_1 (data)
        ts, delta_t = FlowMatching.get_ts_and_delta_t(
            T, t_threshold=t_threshold, adaptive=adaptive
        )

        # Precompute lower bound and upper bound for the repair method
        if (
            task.xl is not None
            and task.xu is not None
            and (isinstance(task.problem, DTLZ) or isinstance(task, SyntheticProblem))
        ):
            xl = task.xl
            xu = task.xu
            if task.is_discrete:
                xl = task.to_logits(np.int64(xl).reshape(1, -1))
                _, dim, n_classes = tuple(xl.shape)
                xl = xl.reshape(-1, dim * n_classes)
            if task.is_sequence:
                xl = task.to_logits(np.int64(xl).reshape(1, -1))
            # shape: (batch_size * O, D)
            xl = task.normalize_x(xl)
            # xl = np.zeros((1, self.D))
            # shape: (D)
            xl = torch.from_numpy(
                xl
            ).squeeze()  # Add squeeze to remove the first dimension of size 1 when its a discrete task or sequence task
            # shape: (batch_size * O, D)
            xl = xl.unsqueeze(0).repeat(batch_size * O, 1)
            # shape: (batch_size * O, D)
            xl = xl.to(device).type(torch.float32)
            if task.is_discrete:
                xu = task.to_logits(np.int64(xu).reshape(1, -1))
                _, dim, n_classes = tuple(xu.shape)
                xu = xu.reshape(-1, dim * n_classes)
            if task.is_sequence:
                xu = task.to_logits(np.int64(xu).reshape(1, -1))
            # shape: (batch_size * O, D)
            xu = task.normalize_x(xu)
            # xu = np.ones((1, self.D))
            # shape: (D)
            xu = torch.from_numpy(
                xu
            ).squeeze()  # Add squeeze to remove the first dimension of size 1 when its a discrete task or sequence task
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
            x_t = self.sample_base(torch.empty(batch_size, self.D)).to(device)

            # Euler method
            count = 0
            for t in ts[1:]:
                count += 1
                # this is x_t + v(x_t, t, y) * delta_t
                # shape: (batch_size, D)
                x_t = x_t + self.weighted_conditional_vnet(
                    x_t,
                    t - delta_t(t),
                    classifiers,
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
                        classifiers,
                        batch_diverse_samples,
                        need_repair=True,
                        xl=xl,
                        xu=xu,
                        task=task,
                    )
                else:
                    scores, merged_samples_x_1 = self.calculate_scores(
                        batch_size, t, O, classifiers, batch_diverse_samples, task=task
                    )

                # Calculate the scores for the neighborhood samples
                # Filter the samples to avoid non-convexity as eq 11 in the paper

                # shape: (batch_size, K, O, len(classifiers))
                neighborhood_scores = scores[neighborhood_indices]
                # shape: (batch_size, K * O, len(classifiers))
                neighborhood_scores = neighborhood_scores.view(
                    batch_size, K * O, len(classifiers)
                )

                # Calculate the angles between the predicted scores and the i-th objective weights
                # shape: (batch_size, K * O, 1)
                angles = FlowMatching.calculate_angles(
                    neighborhood_scores.view(batch_size * K * O, len(classifiers)),
                    objectives_weights.repeat_interleave(K * O, dim=0),
                )
                angles = angles.view(batch_size, K * O, 1)

                # Filter out the samples whose angles are larger than phi_i, find the indices
                # shape: (batch_size, K * O, 1)
                angle_filter_mask = FlowMatching.get_angle_filter_mask(
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

                # Update the pareto set. If the new offspring is better than the i-th solution in the pareto set,
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
        temp_pareto_set = FlowMatching.remove_duplicates(temp_pareto_set)
        assert (
            len(temp_pareto_set) >= num_solutions
        ), "Error: The number of solutions in the pareto set is less than the number of solutions we want to keep"
        temp_pareto_set = torch.stack(temp_pareto_set, dim=0).squeeze()
        temp_pareto_set = task.denormalize_x(temp_pareto_set.cpu().detach().numpy())
        if task.is_discrete:
            temp_pareto_set = temp_pareto_set.reshape(
                temp_pareto_set.shape[0], task.x.shape[1], -1
            )
            temp_pareto_set = task.to_integers(temp_pareto_set)
        if task.is_sequence:
            temp_pareto_set = task.to_integers(temp_pareto_set)
        nadir_point = task.nadir_point
        # For calculating hypervolume, we use the min-max normalization
        res_x = temp_pareto_set
        res_y = task.predict(res_x)
        # Do a non-dominated sorting to get the pareto set
        res_x, res_y = FlowMatching.get_N_non_dominated_solutions(
            res_x, res_y, num_solutions, classifiers
        )
        # Transpose if necessary. This is an issue induced by problematic tasks from the benchmark
        # After communication with the author of the benchmark, we omit to benchmark these tasks
        # To repeat our experiments, we should not transpose the results
        if res_y.shape[0] != res_x.shape[0]:
            res_y = res_y.T
        visible_masks = np.ones(len(res_y))
        visible_masks[np.where(np.logical_or(np.isinf(res_y), np.isnan(res_y)))[0]] = 0
        visible_masks[np.where(np.logical_or(np.isinf(res_x), np.isnan(res_x)))[0]] = 0
        res_x = res_x[np.where(visible_masks == 1)[0]]
        res_y = res_y[np.where(visible_masks == 1)[0]]

        res_y_75_percent = get_quantile_solutions(res_y, 0.75)
        res_y_50_percent = get_quantile_solutions(res_y, 0.50)
        # To calculate hypervolume, we use the min-max normalization as suggested by the benchmark
        res_y = task.normalize_y(res_y, normalization_method="min-max")
        nadir_point = task.normalize_y(nadir_point, normalization_method="min-max")
        res_y_50_percent = task.normalize_y(
            res_y_50_percent, normalization_method="min-max"
        )
        res_y_75_percent = task.normalize_y(
            res_y_75_percent, normalization_method="min-max"
        )

        _, d_best = task.get_N_non_dominated_solutions(
            N=256, return_x=False, return_y=True
        )

        d_best = task.normalize_y(d_best, normalization_method="min-max")

        d_best_hv = hv(nadir_point, d_best, task_name)
        hv_value = hv(nadir_point, res_y, task_name)
        hv_value_50_percentile = hv(nadir_point, res_y_50_percent, task_name)
        hv_value_75_percentile = hv(nadir_point, res_y_75_percent, task_name)

        # print(f"Pareto Set: {pareto_set}")
        print(f"Hypervolume (100th): {hv_value:4f}")
        print(f"Hypervolume (75th): {hv_value_75_percentile:4f}")
        print(f"Hypervolume (50th): {hv_value_50_percentile:4f}")
        print(f"Hypervolume (D(best)): {d_best_hv:4f}")
        # Save the results
        hv_results = {
            "hypervolume/D(best)": d_best_hv,
            "hypervolume/100th": hv_value,
            "hypervolume/75th": hv_value_75_percentile,
            "hypervolume/50th": hv_value_50_percentile,
        }
        pareto_set = [
            pareto_set[i][0] for i in range(batch_size)
        ]  # shape: (batch_size, D)
        pareto_set = torch.stack(pareto_set, dim=0).squeeze()  # shape: (batch_size, D)
        # convert to numpy array
        pareto_set = pareto_set.cpu().detach().numpy()
        # We return all the pareto set and save them to the file
        # Filter samples later during evaluation if needed
        return pareto_set, hv_results

    def log_prob(self, x_1, reduction="mean"):
        # backward Euler (see Appendix C in Lipman's paper)
        ts = torch.linspace(1.0, 0.0, self.T)
        delta_t = ts[1] - ts[0]

        for t in ts:
            if t == 1.0:
                x_t = x_1 * 1.0
                f_t = 0.0
            else:
                # Calculate phi_t
                t_embedding = self.time_embedding(torch.Tensor([t]).to(x_1.device))
                x_t = x_t - self.vnet(x_t + t_embedding) * delta_t

                # Calculate f_t
                # approximate the divergence using the Hutchinson trace estimator and the autograd
                self.vnet.eval()  # set the vector field net to evaluation

                x = torch.tensor(
                    x_t.data, device=x_1.device
                )  # copy the original data (it doesn't require grads!)
                x.requires_grad = True

                e = torch.randn_like(x).to(x_1.device)  # epsilon ~ Normal(0, I)

                e_grad = torch.autograd.grad(self.vnet(x).sum(), x, create_graph=True)[
                    0
                ]
                e_grad_e = e_grad * e
                f_t = e_grad_e.view(x.shape[0], -1).sum(dim=1)

                self.vnet.eval()  # set the vector field net to train again

        log_p_1 = self.log_p_base(x_t, reduction="sum") - f_t

        if reduction == "mean":
            return log_p_1.mean()
        elif reduction == "sum":
            return log_p_1.sum()
