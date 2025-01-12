import numpy as np
from pymoo.core.problem import Problem

from paretoflow.utils import (
    min_max_denormalize,
    min_max_normalize,
    to_integers,
    to_logits,
    z_score_denormalize,
    z_score_normalize,
)


class Task(Problem):
    """
    This class contains the implementation of the Task class,
    which is used to load the input data and initialize the problem.
    """

    def __init__(
        self,
        task_name: str,
        input_x: np.ndarray,
        input_y: np.ndarray,
        x_lower_bound: np.ndarray,
        x_upper_bound: np.ndarray,
        is_discrete: bool = False,
        normalize_x: bool = True,
        normalize_y: bool = True,
        normalization_method: str = "z_score",
        soft_interpolation: float = 0.6,
        nadir_point: np.ndarray = None,
    ):
        """
        Initialize the Task class
        :param task_name: str: the name of the task
        :param input_x: np.ndarray: the input data with shape (n_samples, n_features) or (n_samples, sequence_length)
        :param input_y: np.ndarray: the input labels with shape (n_samples, n_objectives)
        :param x_lower_bound: np.ndarray: the lower bound of the input data with shape (n_features,) or (sequence_length,)
        :param x_upper_bound: np.ndarray: the upper bound of the input data with shape (n_features,) or (sequence_length,)
        :param is_discrete: bool: whether the input data is discrete
        :param normalize_x: bool: whether to normalize the input data
        :param normalize_y: bool: whether to normalize the input labels
        :param normalization_method: str: the method to normalize the input data, can be "z_score" or "min_max"
        :param soft_interpolation: float: the soft interpolation parameter for to_logits
        :param nadir_point: np.ndarray: the nadir point with shape (n_objectives,)
        """
        super().__init__(n_var=input_x.shape[1], n_obj=input_y.shape[1])
        self.name = task_name
        self.n_dim = input_x.shape[1]
        self.n_obj = input_y.shape[1]

        # Load the input data
        self.name = task_name
        self.input_x = input_x.copy()
        self.input_y = input_y.copy()
        self.xl = x_lower_bound.copy()
        self.xu = x_upper_bound.copy()
        self.is_discrete = is_discrete
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.normalization_method = normalization_method
        self.num_classes_on_each_position = x_upper_bound.copy().tolist()
        self.soft_interpolation = soft_interpolation
        self.nadir_point = nadir_point

        # Handle the discrete features
        if self.is_discrete is not None and self.is_discrete:
            assert (
                self.num_classes_on_each_position is not None
            ), "Number of classes on each position is not provided."
            self.input_x = self.to_logits_x(
                self.input_x, self.num_classes_on_each_position, self.soft_interpolation
            )

        # normalize the data
        if self.normalize_x:
            self.input_x, param1, param2 = self.to_normalize_x(
                self.input_x, self.normalization_method
            )
        if self.normalize_y:
            self.input_y, param3, param4 = self.to_normalize_y(
                self.input_y, self.normalization_method
            )
        if self.normalization_method == "z_score":
            self.x_mean = param1
            self.x_std = param2
            self.y_mean = param3
            self.y_std = param4
        elif self.normalization_method == "min_max":
            self.x_min = param1
            self.x_max = param2
            self.y_min = param3
            self.y_max = param4

    def to_normalize_x(self, input_x: np.ndarray, normalization_method: str):
        if normalization_method == "z_score":
            return z_score_normalize(input_x)
        elif normalization_method == "min_max":
            return min_max_normalize(input_x)

    def to_denormalize_x(self, input_x: np.ndarray, normalization_method: str):
        if normalization_method == "z_score":
            return z_score_denormalize(input_x, self.x_mean, self.x_std)
        elif normalization_method == "min_max":
            return min_max_denormalize(input_x, self.x_min, self.x_max)

    def to_normalize_y(self, input_y: np.ndarray, normalization_method: str):
        if normalization_method == "z_score":
            return z_score_normalize(input_y)
        elif normalization_method == "min_max":
            return min_max_normalize(input_y)

    def to_denormalize_y(self, input_y: np.ndarray, normalization_method: str):
        if normalization_method == "z_score":
            return z_score_denormalize(input_y, self.y_mean, self.y_std)
        elif normalization_method == "min_max":
            return min_max_denormalize(input_y, self.y_min, self.y_max)

    def to_logits_x(
        self,
        input_x: np.ndarray,
        num_classes_on_each_position: list,
        soft_interpolation: float,
    ):
        return to_logits(input_x, num_classes_on_each_position, soft_interpolation)

    def to_integers_x(
        self,
        input_x: np.ndarray,
        num_classes_on_each_position: list,
        soft_interpolation: float,
    ):
        return to_integers(input_x, num_classes_on_each_position, soft_interpolation)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("evaluate method is not implemented")
