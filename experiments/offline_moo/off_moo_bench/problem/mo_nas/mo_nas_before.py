import math
import os

import numpy as np
from evoxbench.benchmarks import NASBench201Benchmark

from offline_moo.off_moo_bench.problem.base import BaseProblem

index_to_op_str = {
    0: "nor_conv_3x3",
    1: "avg_pool_3x3",
    2: "nor_conv_1x1",
    3: "skip_connect",
    4: "none",
}


class MO_NAS(BaseProblem):
    def __init__(self, n_obj=3, nadir_point=None, ideal_point=None):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_dim=6,
            n_obj=n_obj,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )

        from evoxbench.database.init import config

        mo_nas_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "m2bo_bench",
            "problem",
            "mo_nas",
        )
        data_path = os.path.join(mo_nas_path, "data")
        database_path = os.path.join(mo_nas_path, "datapath")
        config(data_path=data_path, database_path=database_path)

        self.objs = "err&params&edgegpu_latency"
        self.benchmark = NASBench201Benchmark(
            objs=self.objs, normalized_objectives=False
        )

    def generate_x(self, size):
        return np.random.randint(low=0, high=5, size=(size, self.n_dim))

    def generate_arch_str_from_x(self, x):
        op_str = [index_to_op_str[int(x[i])] for i in range(len(x))]
        return "|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|".format(
            op_str[0], op_str[1], op_str[2], op_str[3], op_str[4], op_str[5]
        )

    def evaluate(self, x):
        assert x.shape[-1] == self.n_dim
        objs = self.benchmark.evaluate(X=x, true_eval=False)
        return objs

    def get_nadir_point(self):
        return np.array([90.288, 1.531546, 11.69270039])

    def get_ideal_point(self):
        return np.array([8.31600001, 0.073306, 0.50138474])


if __name__ == "__main__":
    problem = MO_NAS()
    x = problem.generate_x(size=100)
    print(problem.evaluate(x))
