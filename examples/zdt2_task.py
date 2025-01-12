import numpy as np

from paretoflow import Task


class ZDT2(Task):
    def __init__(self):
        # Load the data
        all_x = np.load("examples/data/zdt2-x-0.npy")
        all_y = np.load("examples/data/zdt2-y-0.npy")
        super().__init__(
            task_name="ZDT2",
            input_x=all_x,
            input_y=all_y,
            x_lower_bound=np.array([0.0] * all_x.shape[1]),
            x_upper_bound=np.array([1.0] * all_x.shape[1]),
            nadir_point=np.array([0.99999706, 9.74316166]),
        )

    def evaluate(self, x):
        """
        This is only for illustrataion purpose, we omit the evaluation function in this example.
        See offline-moo benchmark for more details about the evaluation function for ZDT2.
        Or one can use the `get_problem` function in the `pymoo` package to evaluate the ZDT2 problem.
        """
        pass
