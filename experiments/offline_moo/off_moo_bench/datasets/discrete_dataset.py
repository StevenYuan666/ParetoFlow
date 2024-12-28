import numpy as np

from offline_moo.off_moo_bench.datasets.dataset_builder import DatasetBuilder


def one_hot(a, num_classes):
    """A helper function that converts integers into a floating
    point one-hot representation using pure numpy:
    https://stackoverflow.com/questions/36960320/
    convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy

    """

    out = np.zeros((a.size, num_classes), dtype=np.float32)
    out[np.arange(a.size), a.ravel()] = 1.0
    out.shape = a.shape + (num_classes,)
    return out


class DiscreteDataset(DatasetBuilder):

    name = "DiscreteDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict(
            is_logits=self.is_logits, soft_interpolation=self.soft_interpolation
        )

    @property
    def subclass(self):
        return DiscreteDataset

    def __init__(
        self, *args, is_logits=False, num_classes=2, soft_interpolation=0.6, **kwargs
    ):

        self.soft_interpolation = soft_interpolation
        self.num_classes = num_classes
        self.is_logits = is_logits

        # initialize the dataset using the method in the base class
        super(DiscreteDataset, self).__init__(*args, **kwargs)

    def batch_transform(self, x_batch, y_batch, return_x=True, return_y=True):

        # convert the design values from integers to logits
        if self.is_logits and return_x and np.issubdtype(x_batch.dtype, np.integer):
            x_batch = self.to_logits(x_batch)

        # convert the design values from logits to integers
        elif (
            not self.is_logits
            and return_x
            and np.issubdtype(x_batch.dtype, np.floating)
        ):
            x_batch = self.to_integers(x_batch)

        # return processed batches of designs an predictions
        return super(DiscreteDataset, self).batch_transform(
            x_batch, y_batch, return_x=return_x, return_y=return_y
        )

    def update_x_statistics(self):
        # handle corner case when we need statistics but they were
        # not computed yet and the dataset is currently mapped to integers
        original_is_logits = self.is_logits
        self.is_logits = True
        super(DiscreteDataset, self).update_x_statistics()
        self.is_logits = original_is_logits

    def map_normalize_x(self):
        # check that the dataset is in a form that supports normalization
        if not self.is_logits:
            raise ValueError("cannot normalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_normalize_x()

    def normalize_x(self, x):
        """
        The default implementation of normalize_x in DatasetBuilder requires
        mapping the input to logits first. However, we want to loose this
        requirement for SequenceDataset. Hence we need to override the
        normalize_x method.
        """
        # check that the dataset is in a form that supports normalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot normalize discrete design values")

        if self.x_normalize_method == "z-score":
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            x = (x - x_mean) / x_std
            # Update the recorded mean and std
            self.x_mean = x_mean
            self.x_standard_dev = x_std
        elif self.x_normalize_method == "min-max":
            x_min = np.min(x, axis=0)
            x_max = np.max(x, axis=0)
            x = (x - x_min) / (x_max - x_min)
            # Update the recorded min and max
            self.x_min = x_min
            self.x_max = x_max
        else:
            raise ValueError("Unknown normalization method")
        return x

    def map_denormalize_x(self):
        # check that the dataset is in a form that supports denormalization
        if not self.is_logits:
            raise ValueError("cannot denormalize discrete design values")

        # call the normalization method of the super class
        super(DiscreteDataset, self).map_denormalize_x()

    def denormalize_x(self, x):
        """
        The default implementation of denormalize_x in DatasetBuilder requires
        mapping the input to logits first. However, we want to loose this
        requirement for SequenceDataset. Hence we need to override the
        denormalize_x method.
        """
        # check that the dataset is in a form that supports denormalization
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot denormalize discrete design values")

        if self.x_normalize_method == "z-score":
            x = x * self.x_standard_dev + self.x_mean
        elif self.x_normalize_method == "min-max":
            x = x * (self.x_max - self.x_min) + self.x_min
        else:
            raise ValueError("Unknown normalization method")
        return x

    def to_logits(self, x):

        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        # convert the integers to one hot vectors
        one_hot_x = one_hot(x, self.num_classes)

        # build a uniform distribution to interpolate between
        uniform_prior = np.full_like(one_hot_x, 1 / float(self.num_classes))

        # interpolate between a dirac distribution and a uniform prior
        soft_x = (
            self.soft_interpolation * one_hot_x
            + (1.0 - self.soft_interpolation) * uniform_prior
        )

        # convert to log probabilities
        x = np.log(soft_x)

        # remove one degree of freedom caused by \sum_i p_i = 1.0
        return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

    def to_integers(self, x):

        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        # add an additional component of zero and find the class
        # with maximum probability
        return np.argmax(
            np.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[1, 0]]), axis=-1
        ).astype(np.int32)

    def map_to_logits(self):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if not self.is_logits:

            # set the appropriate state variable
            self.is_logits = True

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break

    def map_to_integers(self):

        # check that statistics are not frozen for this dataset
        if self.freeze_statistics:
            raise ValueError("cannot update dataset when it is frozen")

        # check design values and prediction values are not normalized
        if self.is_logits:

            # if design values are normalized then denorm
            if self.is_normalized_x:
                self.map_denormalize_x()

            # set the appropriate state variable
            self.is_logits = False

            # check shape and data type of a single design value x
            for x0 in self.iterate_samples(return_y=False):
                self.input_shape = x0.shape
                self.input_size = int(np.prod(x0.shape))
                self.input_dtype = x0.dtype
                break

            for x0 in self.iterate_test_samples(return_y=False):
                assert self.input_shape == x0.shape
                assert self.input_size == int(np.prod(x0.shape))
                assert self.input_dtype == x0.dtype
                break
