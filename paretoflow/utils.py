import numpy as np


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


def help_to_logits(x: np.ndarray, num_classes: int, soft_interpolation: float = 0.6):
    """
    Convert the integers to logits.
    :param x: np.ndarray: the input data, with shape (n_samples, 1)
    :param num_classes: int: the number of classes
    :param soft_interpolation: float: the soft interpolation parameter
    :return: np.ndarray: the logits, with shape (n_samples, 1, n_classes)
    """
    # check that the input format is correct
    if not np.issubdtype(x.dtype, np.integer):
        raise ValueError("cannot convert non-integers to logits")

    # convert the integers to one hot vectors
    one_hot_x = one_hot(x, num_classes)

    # build a uniform distribution to interpolate between
    uniform_prior = np.full_like(one_hot_x, 1 / float(num_classes))

    # interpolate between a dirac distribution and a uniform prior
    soft_x = soft_interpolation * one_hot_x + (1.0 - soft_interpolation) * uniform_prior

    # convert to log probabilities
    x = np.log(soft_x)

    # remove one degree of freedom caused by \sum_i p_i = 1.0
    return (x[:, :, 1:] - x[:, :, :1]).astype(np.float32)


def to_logits(
    x: np.ndarray,
    num_classes_on_each_position: list[int],
    soft_interpolation: float = 0.6,
):
    """
    Since this is a sequence of categorical variables, we need to
    convert the integers at each position to logits. Then we will
    concatenate the logits for each position to get the final logits
    :param x: np.ndarray: the input data, with shape (n_samples, seq_len)
    :param num_classes_on_each_position: list[int]: the number of classes on each position
    :param soft_interpolation: float: the soft interpolation parameter
    :return: np.ndarray: the logits, with shape (n_samples, seq_len, n_classes)
    """
    num_classes = num_classes_on_each_position
    num_classes = [i + 1 for i in num_classes]
    # if count[i] = 1, then we will have a single class, so we add 1
    # to the such cases, this is only happening for RFP-Exact-v0
    # treat this as adding a dummy class
    num_classes = [i + 1 if i == 1 else i for i in num_classes]
    logits = []
    sequence_length = len(num_classes)
    for i in range(sequence_length):
        temp_x = x[:, i].reshape(-1, 1)
        logits.append(help_to_logits(temp_x, num_classes[i], soft_interpolation))
    return np.concatenate(logits, axis=2).squeeze()


def help_to_integers(x: np.ndarray, true_num_of_classes: int):
    """
    Convert the logits to integers.
    :param x: np.ndarray: the input data, with shape (n_samples, 1, n_classes)
    :param true_num_of_classes: int: the true number of classes
    :return: np.ndarray: the integers, with shape (n_samples, 1)
    """
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


def to_integers(x: np.ndarray, num_classes_on_each_position: list[int]):
    """
    Since this is a sequence of categorical variables, we need to
    convert the logits at each position to integers. Then we will
    concatenate the integers for each position to get the final integers
    :param x: np.ndarray: the input data, with shape (n_samples, seq_len, n_classes)
    :param num_classes_on_each_position: list[int]: the number of classes on each position
    :return: np.ndarray: the integers, with shape (n_samples, seq_len)
    """
    true_num_classes = num_classes_on_each_position
    true_num_classes = [i + 1 for i in true_num_classes]
    # if count[i] = 1, then we will have a single class, so we
    # add 1 to the such cases, this is only happening for RFP-Exact-v0
    # treat this as adding a dummy class
    num_classes = [i + 1 if i == 1 else i for i in true_num_classes]
    integers = []
    start = 0
    sequence_length = len(num_classes)
    for i in range(sequence_length):
        temp_x = x[:, start : (start + num_classes[i] - 1)].reshape(
            -1, 1, num_classes[i] - 1
        )
        integers.append(help_to_integers(temp_x, num_classes[i]))
        start += num_classes[i] - 1
    return np.concatenate(integers, axis=1)


def z_score_normalize_x(x: np.ndarray):
    """
    This function is used to normalize the input data using z-score normalization.
    :param x: np.ndarray: the input data, with shape (n_samples, n_dim)
    :return: np.ndarray: the normalized data, with shape (n_samples, n_dim),
    :return: np.ndarray: the mean of the data, with shape (n_dim,)
    :return: np.ndarray: the standard deviation of the data, with shape (n_dim,)
    """
    # check that the dataset is in a form that supports normalization
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("cannot normalize discrete design values")

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std
    # Update the recorded mean and std
    return x, x_mean, x_std


def z_score_denormalize_x(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    """
    This function is used to denormalize the input data using z-score denormalization.
    :param x: np.ndarray: the input data, with shape (n_samples, n_dim)
    :param x_mean: np.ndarray: the mean of the data, with shape (n_dim,)
    :param x_std: np.ndarray: the standard deviation of the data, with shape (n_dim,)
    :return: np.ndarray: the denormalized data, with shape (n_samples, n_dim)
    """
    # check that the dataset is in a form that supports denormalization
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError("cannot denormalize discrete design values")

    x = x * x_std + x_mean
    return x


def min_max_normalize_x(x: np.ndarray):
    """
    This function is used to normalize the input data using min-max normalization.
    :param x: np.ndarray: the input data, with shape (n_samples, n_dim)
    :return: np.ndarray: the normalized data, with shape (n_samples, n_dim),
    :return: np.ndarray: the minimum of the data, with shape (n_dim,)
    :return: np.ndarray: the maximum of the data, with shape (n_dim,)
    """
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min)
    return x, x_min, x_max


def min_max_denormalize_x(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray):
    """
    This function is used to denormalize the input data using min-max denormalization.
    :param x: np.ndarray: the input data, with shape (n_samples, n_dim)
    :param x_min: np.ndarray: the minimum of the data, with shape (n_dim,)
    :param x_max: np.ndarray: the maximum of the data, with shape (n_dim,)
    :return: np.ndarray: the denormalized data, with shape (n_samples, n_dim)
    """
    x = x * (x_max - x_min) + x_min
    return x
