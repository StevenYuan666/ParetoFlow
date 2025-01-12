import importlib
import re
from typing import Union

from offline_moo.off_moo_bench.datasets.dataset_builder import DatasetBuilder
from offline_moo.off_moo_bench.datasets.discrete_dataset import DiscreteDataset
from offline_moo.off_moo_bench.datasets.sequence_dataset import SequenceDataset
from offline_moo.off_moo_bench.disk_resource import DiskResource
from offline_moo.off_moo_bench.evaluation.metrics import hv
from offline_moo.off_moo_bench.problem.base import BaseProblem

# used to determine the name of a dataset that is sharded to disk
SHARD_PATTERN = re.compile(r"(.+)-(\w)-(\d+).npy$")


# this is used to import data set classes dynamically
def import_name(name):
    mod_name, attr_name = name.split(":")
    return getattr(importlib.import_module("offline_moo." + mod_name), attr_name)


class Task(object):

    def __init__(
        self,
        dataset: Union[DatasetBuilder, type, str],
        problem: Union[BaseProblem, type, str],
        dataset_kwargs=None,
        problem_kwargs=None,
        relabel=False,
    ):

        # use additional_kwargs to override self.kwargs
        kwargs = dataset_kwargs if dataset_kwargs else dict()

        # if self.entry_point is a function call it
        if callable(dataset):
            dataset = dataset(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(dataset, str):
            dataset = import_name(dataset)(**kwargs)

        # return if the dataset could not be loaded
        elif not isinstance(dataset, DatasetBuilder):
            raise ValueError("dataset could not be loaded")

        # expose the built dataset
        self.dataset = dataset

        # use additional_kwargs to override self.kwargs
        kwargs = problem_kwargs if problem_kwargs else dict()

        # if self.entry_point is a function call it
        if callable(problem):
            problem = problem(**kwargs)

        # if self.entry_point is a string import it first
        elif isinstance(problem, str):
            problem = import_name(problem)(**kwargs)

        # return if the oracle could not be loaded
        elif not isinstance(problem, BaseProblem):
            raise ValueError("problem could not be loaded")

        # expose the built oracle
        self.problem = problem

        # only relabel when an approximate model is used
        relabel = relabel and isinstance(problem, BaseProblem)

        # attempt to download the appropriate shards
        new_shards = []
        for shard in dataset.y_shards:
            if relabel and isinstance(shard, DiskResource):

                m = SHARD_PATTERN.search(shard.disk_target)
                file = f"{m.group(1)}-{problem.name}-y-{m.group(3)}.npy"

                # create a disk resource for the new shard
                new_shards.append(
                    DiskResource(
                        file,
                        is_absolute=False,
                    )
                )

        # check if every shard was downloaded successfully
        # this naturally handles when the shard is already downloaded
        if (
            relabel
            and len(new_shards) > 0
            and all([f.is_downloaded for f in new_shards])
        ):

            # assign the y shards to the downloaded files and re sample
            # the dataset if sub sampling is being used
            dataset.y_shards = new_shards
            dataset.subsample(
                max_samples=dataset.dataset_size,
                max_percentile=dataset.dataset_max_percentile,
                min_percentile=dataset.dataset_min_percentile,
            )

        elif relabel:

            # test if the shards are stored on the disk
            # this means that downloading cached predictions failed
            name = None
            test_shard = dataset.y_shards[0]
            if isinstance(test_shard, DiskResource):

                # create a name for the new sharded prediction
                m = SHARD_PATTERN.search(test_shard.disk_target)
                name = f"{m.group(1)}-{problem.name}"

            # relabel the dataset using the new oracle model
            dataset.relabel(
                lambda x, y: problem.evaluate(x),
                to_disk=name is not None,
                is_absolute=False,
                disk_target=name,
            )

        if isinstance(self.dataset, SequenceDataset):
            # Change to numpy int
            classes = self.xu.astype(int) + 1
            self.dataset.num_classes_on_each_dim = classes.tolist()

    @property
    def is_discrete(self):
        """Attribute that specifies whether the task dataset is discrete or
        continuous determined by whether the dataset instance task.dataset
        inherits from DiscreteDataset or ContinuousDataset

        """

        return isinstance(self.dataset, DiscreteDataset)

    @property
    def is_sequence(self):
        """Attribute that specifies whether the task dataset is sequence or not
        determined by whether the dataset instance task.dataset
        inherits from SequenceDataset or not

        """

        return isinstance(self.dataset, SequenceDataset)

    @property
    def problem_name(self):
        """Attribute that specifies the name of a machine learning model used
        as the ground truth score function for a model-based optimization
        problem; for example, "random_forest"

        """

        return self.problem.name

    @property
    def dataset_name(self):
        """An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench

        """

        return self.dataset.name

    @property
    def x_name(self):
        """An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

        """

        return self.dataset.x_name

    @property
    def y_name(self):
        """An attribute that specifies the name of predictions in a
        model-based optimization dataset, which might be used when labelling
        plots in a visualization of performance in a research paper

        """

        return self.dataset.y_name

    @property
    def dataset_size(self):
        """the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer

        """

        return self.dataset.dataset_size

    @property
    def dataset_distribution(self):
        """the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution

        """

        return self.dataset.dataset_distribution

    @property
    def dataset_max_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_max_percentile

    @property
    def dataset_min_percentile(self):
        """the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class

        """

        return self.dataset.dataset_min_percentile

    @property
    def input_shape(self):
        """the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape

        """

        return self.dataset.input_shape

    @property
    def input_size(self):
        """the total number of components in the design values 'x',
        represented as a single integer, the product of its shape entries

        """

        return self.dataset.input_size

    @property
    def input_dtype(self):
        """the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

        """

        return self.dataset.input_dtype

    @property
    def output_shape(self):
        """the shape of a single prediction value 'y', represented as a list
        of integers similar to calling np.ndarray.shape

        """

        return self.dataset.output_shape

    @property
    def output_size(self):
        """the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries

        """

        return self.dataset.output_size

    @property
    def output_dtype(self):
        """the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

        """

        return self.dataset.output_dtype

    @property
    def x(self):
        """the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type

        """

        return self.dataset.x

    @property
    def y(self):
        """the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'

        """

        return self.dataset.y

    @property
    def x_test(self):
        """the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type

        """

        return self.dataset.x_test

    @property
    def y_test(self):
        """the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'

        """

        return self.dataset.y_test

    @property
    def is_normalized_x(self):
        """a boolean indicator that specifies whether the design values in
        the dataset are being normalized

        """

        return self.dataset.is_normalized_x

    @property
    def is_normalized_y(self):
        """a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized

        """

        return self.dataset.is_normalized_y

    @property
    def forbidden_normalize_x(self):
        return self.dataset.forbidden_normalize_x

    @property
    def xl(self):
        return self.problem.xl

    @property
    def xu(self):
        return self.problem.xu

    @property
    def nadir_point(self):
        return self.problem.get_nadir_point()

    @property
    def is_logits(self):
        """a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

        """

        if not hasattr(self.dataset, "is_logits"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.is_logits

    @property
    def num_classes(self):
        """an integer representing the number of classes in the distribution
        that the integer data points are sampled from which cannot be None
        and must also be greater than 1

        """

        if not hasattr(self.dataset, "num_classes"):
            raise ValueError("only supported on discrete datasets")
        return self.dataset.num_classes

    def iterate_batches(
        self, batch_size, return_x=True, return_y=True, drop_remainder=False
    ):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        batch_size: int
            a positive integer that specifies the batch size of samples
            taken from a model-based optimization data set; batches
            with batch_size elements are yielded
        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True
        drop_remainder: bool
            a boolean indicator representing whether the last batch
            should be dropped in the case it has fewer than batch_size
            elements; the default behavior is not to drop the smaller batch.

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(
            self.dataset.iterate_batches(
                batch_size,
                return_x=return_x,
                return_y=return_y,
                drop_remainder=drop_remainder,
            )
        )

    def iterate_samples(self, return_x=True, return_y=True):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Arguments:

        return_x: bool
            a boolean indicator that specifies whether the generator yields
            design values at every iteration; note that at least one of
            return_x and return_y must be set to True
        return_y: bool
            a boolean indicator that specifies whether the generator yields
            prediction values at every iteration; note that at least one
            of return_x and return_y must be set to True

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.dataset.iterate_samples(return_x=return_x, return_y=return_y))

    def __iter__(self):
        """Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

        Returns:

        generator: Iterator
            a python iterable that yields samples from a model-based
            optimization data set and returns once finished

        """

        return iter(self.dataset)

    def map_normalize_x(self):
        """a function that standardizes the design values 'x' to have zero
        empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_normalize_x()

    def map_normalize_y(self):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_normalize_y()

    def map_denormalize_x(self):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_denormalize_x()

    def map_denormalize_y(self):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance in the dataset

        """

        self.dataset.map_denormalize_y()

    def map_to_integers(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

        """

        if not hasattr(self.dataset, "map_to_integers"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_integers()

    def map_to_logits(self):
        """a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits

        """

        if not hasattr(self.dataset, "map_to_logits"):
            raise ValueError("only supported on discrete datasets")
        self.dataset.map_to_logits()

    def get_N_non_dominated_solutions(self, *args, **kwargs):
        return self.dataset.get_N_non_dominated_solutions(*args, **kwargs)

    def normalize_x(self, x, normalization_method="z-score"):
        """a function that standardizes the design values 'x' to have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be normalized according to dataset statistics

        normalization_method: str
            a string that specifies the normalization method to use
            "min-max" for min-max normalization
            "z-score" for z-score normalization

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been normalized using dataset statistics

        """
        self.dataset.x_normalize_method = normalization_method
        return self.dataset.normalize_x(x)

    def normalize_y(self, y, normalization_method="z-score"):
        """a function that standardizes the prediction values 'y' to have
        zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be normalized according to dataset statistics

        normalization_method: str
            a string that specifies the normalization method to use
            "min-max" for min-max normalization
            "z-score" for z-score normalization

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been normalized using dataset statistics

        """
        self.dataset.y_normalize_method = normalization_method
        return self.dataset.normalize_y(y)

    def denormalize_x(self, x, normalization_method="z-score"):
        """a function that un-standardizes the design values 'x' which have
        zero empirical mean and unit empirical variance

        Arguments:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            shall be denormalized according to dataset statistics

        normalization_method: str
            a string that specifies the normalization method to use
            "min-max" for min-max normalization
            "z-score" for z-score normalization

        Returns:

        x: np.ndarray
            a design value represented as a numpy array potentially
            given as a batch of designs which
            has been denormalized using dataset statistics

        """
        self.dataset.x_normalize_method = normalization_method
        return self.dataset.denormalize_x(x)

    def denormalize_y(self, y, normalization_method="z-score"):
        """a function that un-standardizes the prediction values 'y' which
        have zero empirical mean and unit empirical variance

        Arguments:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            shall be denormalized according to dataset statistics

        normalization_method: str
            a string that specifies the normalization method to use
            "min-max" for min-max normalization
            "z-score" for z-score normalization

        Returns:

        y: np.ndarray
            a prediction value represented as a numpy array potentially
            given as a batch of predictions which
            has been denormalized using dataset statistics

        """
        self.dataset.y_normalize_method = normalization_method
        return self.dataset.denormalize_y(y)

    def to_integers(self, x):
        """A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which have been converted from a floating point
            representation of a certain probability distribution

        """

        if not hasattr(self.dataset, "to_integers"):
            raise ValueError(
                "only supported on discrete datasets and sequence datasets"
            )
        return self.dataset.to_integers(x)

    def to_logits(self, x):
        """A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution

        Arguments:

        x: np.ndarray
            a numpy array containing design values represented as integers
            which are going to be converted into a floating point
            representation of a certain probability distribution

        Returns:

        x: np.ndarray
            a numpy array containing design values represented as floating
            point numbers which have be converted from integer samples of
            a certain probability distribution

        """

        if not hasattr(self.dataset, "to_logits"):
            raise ValueError(
                "only supported on discrete datasets and sequence datasets"
            )
        return self.dataset.to_logits(x)

    def predict(self, x_batch, **kwargs):
        """a function that accepts a batch of design values 'x' as input and
        for each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

        Arguments:

        x_batch: np.ndarray
            a batch of design values 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch of prediction values 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        if self.is_discrete and self.is_logits:
            if len(x_batch.shape) == 2:
                x_batch = x_batch.reshape(
                    x_batch.shape[0], -1, self.dataset.num_classes - 1
                )

        if not self.forbidden_normalize_x:
            if self.problem.requires_normalized_x and not self.is_normalized_x:
                x_batch = self.normalize_x(x_batch, normalization_method="min-max")
            elif not self.problem.requires_normalized_x and self.is_normalized_x:
                x_batch = self.denormalize_x(x_batch, normalization_method="min-max")

        if self.is_discrete and self.is_logits:
            self.map_to_integers()
            x_batch = self.to_integers(x_batch)
            self.map_to_logits()

        return self.problem.evaluate(x_batch, **kwargs)
