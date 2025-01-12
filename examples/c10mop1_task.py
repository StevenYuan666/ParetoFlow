import numpy as np
from pymoo.problems import get_problem

from paretoflow import Task


class C10MOP1(Task):
    def __init__(self):
        # Load the data
        all_x = np.load("examples/data/c10mop1-x-0.npy")
        all_y = np.load("examples/data/c10mop1-y-0.npy")
        super().__init__(
            task_name="C10MOP1",
            input_x=all_x,
            input_y=all_y,
            x_lower_bound=np.array([0.0] * all_x.shape[1]),
            x_upper_bound=np.array(
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                ]
            ),
            is_discrete=True,
            nadir_point=np.array(
                [3.49e-1, 3.14e7]
            ),  # This is the nadir point for C10MOP1 from offline-moo benchmark
        )

    def evaluate(self, x):
        """
        We omit the evaluation function in this example, as this implementation requires the NASBench101Benchmark.
        See offline-moo benchmark for more details about the evaluation function for C10MOP1.
        """
        pass
