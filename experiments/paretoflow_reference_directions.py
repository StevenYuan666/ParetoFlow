import numpy as np
from scipy import special


# the newest version of pymoo is not compatible with the code
class ReferenceDirectionFactory:

    def __init__(
        self, n_dim, scaling=None, lexsort=True, verbose=False, seed=None, **kwargs
    ) -> None:
        super().__init__()
        self.n_dim = n_dim
        self.scaling = scaling
        self.lexsort = lexsort
        self.verbose = verbose
        self.seed = seed

    def __call__(self):
        return self.do()

    def do(self):

        # set the random seed if it is provided
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.n_dim == 1:
            return np.array([[1.0]])
        else:

            val = self._do()
            if isinstance(val, tuple):
                ref_dirs, other = val[0], val[1:]
            else:
                ref_dirs = val

            if self.scaling is not None:
                ref_dirs = scale_reference_directions(ref_dirs, self.scaling)

            # do ref_dirs is desired
            if self.lexsort:
                I = np.lexsort([ref_dirs[:, j] for j in range(ref_dirs.shape[1])][::-1])
                ref_dirs = ref_dirs[I]

            return ref_dirs

    def _do(self):
        return None


# =========================================================================================================
# Das Dennis Reference Directions (Uniform)
# =========================================================================================================
def scale_reference_directions(ref_dirs, scaling):
    return ref_dirs * scaling + ((1 - scaling) / ref_dirs.shape[1])


def get_number_of_uniform_points(n_partitions, n_dim):
    """
    Returns the number of uniform points that can be created uniformly.
    """
    return int(special.binom(n_dim + n_partitions - 1, n_partitions))


def get_partition_closest_to_points(n_points, n_dim):
    """
    Returns the corresponding partition number which create the desired number of points
    or less!
    """

    if n_dim == 1:
        return 0

    n_partitions = 1
    _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    while _n_points <= n_points:
        n_partitions += 1
        _n_points = get_number_of_uniform_points(n_partitions, n_dim)
    return n_partitions - 1


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(
                ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1
            )


class UniformReferenceDirectionFactory(ReferenceDirectionFactory):

    def __init__(
        self, n_dim, scaling=None, n_points=None, n_partitions=None, **kwargs
    ) -> None:
        super().__init__(n_dim, scaling=scaling, **kwargs)

        if n_points is not None:
            n_partitions = get_partition_closest_to_points(n_points, n_dim)
            results_in = get_number_of_uniform_points(n_partitions, n_dim)

            # the number of points are not matching to any partition number
            if results_in != n_points:
                results_in_next = get_number_of_uniform_points(n_partitions + 1, n_dim)
                raise Exception(
                    "The number of points (n_points = %s) can not be created uniformly.\n"
                    "Either choose n_points = %s (n_partitions = %s) or "
                    "n_points = %s (n_partitions = %s)."
                    % (
                        n_points,
                        results_in,
                        n_partitions,
                        results_in_next,
                        n_partitions + 1,
                    )
                )

            self.n_partitions = n_partitions

        elif n_partitions is not None:
            self.n_partitions = n_partitions

        else:
            raise Exception("Either provide number of partitions or number of points.")

    def _do(self):
        return das_dennis(self.n_partitions, self.n_dim)
