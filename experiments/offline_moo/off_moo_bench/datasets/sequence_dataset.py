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


class SequenceDataset(DatasetBuilder):

    name = "SequenceDataset"
    x_name = "Design"
    y_name = "Prediction"

    @property
    def subclass_kwargs(self):
        return dict(forbidden_normalize_x=self.forbidden_normalize_x)

    @property
    def subclass(self):
        return SequenceDataset

    def __init__(self, *args, **kwargs):
        super(SequenceDataset, self).__init__(
            forbidden_normalize_x=False, *args, **kwargs
        )
        self.soft_interpolation = 0.6
        self.num_classes_on_each_dim = None

    # We should not use this property, because the number of classes is given by the task xu, i.e., the upperbound
    # See task.py line 126 - line 129

    # @property
    # def num_classes_on_each_dim(self):
    #     count = {}
    #     for i in range(self.x.shape[1]):
    #         # the maximum value of the i-th dimension
    #         count[i] = (
    #             np.max(self.x[:, i]) + 1
    #         )  # I initially thought it was np.unique(self.x[:, i]), but seems like some classes are not present in the dataset
    #     count = list(count.values())
    #     return count

    @property
    def sequence_length(self):
        return self.x.shape[1]

    def help_to_logits(self, x, num_classes):
        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.integer):
            raise ValueError("cannot convert non-integers to logits")

        # convert the integers to one hot vectors
        one_hot_x = one_hot(x, num_classes)

        # build a uniform distribution to interpolate between
        uniform_prior = np.full_like(one_hot_x, 1 / float(num_classes))

        # interpolate between a dirac distribution and a uniform prior
        soft_x = (
            self.soft_interpolation * one_hot_x
            + (1.0 - self.soft_interpolation) * uniform_prior
        )

        # convert to log probabilities
        x = np.log(soft_x)

        # remove one degree of freedom caused by \sum_i p_i = 1.0
        return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)

    def to_logits(self, x):
        """
        Since this is a sequence of categorical variables, we need to
        convert the integers at each position to logits. Then we will
        concatenate the logits for each position to get the final logits
        """
        num_classes = self.num_classes_on_each_dim
        # if count[i] = 1, then we will have a single class, so we add 1 to the such cases, this is only happening for RFP-Exact-v0
        # treat this as adding a dummy class
        num_classes = [i + 1 if i == 1 else i for i in num_classes]
        logits = []
        for i in range(self.sequence_length):
            temp_x = x[:, i].reshape(-1, 1)
            logits.append(self.help_to_logits(temp_x, num_classes[i]))
        return np.concatenate(logits, axis=2).squeeze()

    def help_to_integers(self, x, true_num_of_classes):

        # check that the input format is correct
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("cannot convert non-floats to integers")

        # Since we might add a dummy class, we need to make sure that the last class is not selected
        # For RFP-Exact-v0
        if true_num_of_classes == 1:
            return np.zeros(x.shape[:-1], dtype=np.int32)

        # add an additional component of zero and find the class
        # with maximum probability
        return np.argmax(
            np.pad(x, [[0, 0]] * (len(x.shape) - 1) + [[1, 0]]), axis=-1
        ).astype(np.int32)

    def to_integers(self, x):
        """
        Since this is a sequence of categorical variables, we need to
        convert the logits at each position to integers. Then we will
        concatenate the integers for each position to get the final integers
        """
        true_num_classes = self.num_classes_on_each_dim
        # if count[i] = 1, then we will have a single class, so we add 1 to the such cases, this is only happening for RFP-Exact-v0
        # treat this as adding a dummy class
        num_classes = [i + 1 if i == 1 else i for i in true_num_classes]
        integers = []
        start = 0
        for i in range(self.sequence_length):
            temp_x = x[:, start : (start + num_classes[i] - 1)].reshape(
                -1, 1, num_classes[i] - 1
            )
            integers.append(self.help_to_integers(temp_x, num_classes[i]))
            start += num_classes[i] - 1
        return np.concatenate(integers, axis=1)

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
