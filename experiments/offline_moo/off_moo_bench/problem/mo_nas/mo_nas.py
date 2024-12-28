import math
import os

import numpy as np

# from evoxbench.benchmarks import NASBench201Benchmark
# from evoxbench.test_suites import c10mop
# from evoxbench.test_suites import in1kmop
from evoxbench.benchmarks import (
    DARTSBenchmark,
    MobileNetV3Benchmark,
    NASBench101Benchmark,
    NASBench201Benchmark,
    NATSBenchmark,
    ResNet50DBenchmark,
    TransformerBenchmark,
)

# from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from offline_moo.off_moo_bench.problem.base import BaseProblem

# from evoxbench.database.init import config
# mo_nas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2bo_bench", "problem", "mo_nas")
# data_path = os.path.join(mo_nas_path, "data")
# database_path = os.path.join(mo_nas_path, "datapath")
# config(data_path=data_path, database_path=database_path)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)


class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        """

        Returns
        -------
        object
        """
        super().__init__(**kwargs)

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        X = np.around(X).astype(int)
        pop.set("X", X)
        return pop


def c10mop(problem_id):
    if problem_id == 1:
        return NASBench101Benchmark(objs="err&params", normalized_objectives=False)
    elif problem_id == 2:
        return NASBench101Benchmark(
            objs="err&params&flops", normalized_objectives=False
        )
    elif problem_id == 3:
        return NATSBenchmark(
            90, objs="err&params&flops", dataset="cifar10", normalized_objectives=False
        )
    elif problem_id == 4:
        return NATSBenchmark(
            90,
            objs="err&params&flops&latency",
            dataset="cifar10",
            normalized_objectives=False,
        )
    elif problem_id == 5:
        return NASBench201Benchmark(
            200,
            objs="err&params&flops&edgegpu_latency&edgegpu_energy",
            dataset="cifar10",
            normalized_objectives=False,
        )
    elif problem_id == 6:
        return NASBench201Benchmark(
            200,
            objs="err&params&flops&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity",
            dataset="cifar10",
            normalized_objectives=False,
        )
    elif problem_id == 7:
        return NASBench201Benchmark(
            200,
            objs="err&params&flops&edgegpu_latency&edgegpu_energy"
            "&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity",
            dataset="cifar10",
            normalized_objectives=False,
        )
    elif problem_id == 8:
        return DARTSBenchmark(objs="err&params", normalized_objectives=False)
    elif problem_id == 9:
        return DARTSBenchmark(objs="err&params&flops", normalized_objectives=False)
    else:
        raise ValueError("the requested problem id does not exist")


def in1kmop(problem_id):
    if problem_id == 1:
        return ResNet50DBenchmark(objs="err&params", normalized_objectives=False)
    elif problem_id == 2:
        return ResNet50DBenchmark(objs="err&flops", normalized_objectives=False)
    elif problem_id == 3:
        return ResNet50DBenchmark(objs="err&params&flops", normalized_objectives=False)
    elif problem_id == 4:
        return TransformerBenchmark(objs="err&params", normalized_objectives=False)
    elif problem_id == 5:
        return TransformerBenchmark(objs="err&flops", normalized_objectives=False)
    elif problem_id == 6:
        return TransformerBenchmark(
            objs="err&params&flops", normalized_objectives=False
        )
    elif problem_id == 7:
        return MobileNetV3Benchmark(objs="err&params", normalized_objectives=False)
    elif problem_id == 8:
        return MobileNetV3Benchmark(
            objs="err&params&flops", normalized_objectives=False
        )
    elif problem_id == 9:
        return MobileNetV3Benchmark(
            objs="err&params&flops&latency", normalized_objectives=False
        )
    else:
        raise ValueError("the requested problem id does not exist")


def get_genetic_operator(
    crx_prob=1.0,  # crossover probability
    crx_eta=30.0,  # SBX crossover eta
    mut_prob=0.9,  # mutation probability
    mut_eta=20.0,  # polynomial mutation hyperparameter eta
):
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=crx_prob, eta=crx_eta)
    mutation = PM(prob=mut_prob, eta=mut_eta)
    repair = RoundingRepair()
    return sampling, crossover, mutation, repair


class MONASProblem(BaseProblem):
    def __init__(self, benchmark, nadir_point=None, ideal_point=None, **kwargs):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_dim=benchmark.search_space.n_var,
            n_obj=benchmark.evaluator.n_objs,
            xl=benchmark.search_space.lb,
            xu=benchmark.search_space.ub,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
            type_var=np.int64,
            **kwargs
        )

        self.benchmark = benchmark

    def _evaluate(self, x, out, *args, **kwargs):

        F = self.benchmark.evaluate(x, true_eval=False)

        out["F"] = F

    def get_nadir_point(self):
        return self.nadir_point

    def get_ideal_point(self):
        return self.ideal_point


class C10MOP1(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(1),
            nadir_point=[3.49158645e-01, 3.13890660e07],
            ideal_point=[5.27844429e-02, 3.01074000e05],
        )


class C10MOP2(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(2),
            nadir_point=[9.04947914e-01, 3.05153380e07, 8.96841524e09],
            ideal_point=[5.27844429e-02, 4.17813000e05, 1.35138890e08],
        )


class C10MOP3(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(3),
            nadir_point=[23.11200001, 0.713674, 274.399882],
            ideal_point=[9.28800003, 0.011714, 4.481114],
        )


class C10MOP4(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(4),
            nadir_point=[2.31120000e01, 7.13674000e-01, 2.74399882e02, 2.11952626e-02],
            ideal_point=[9.28800003, 0.011714, 4.481114, 0.0128115],
        )


class C10MOP5(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(5),
            nadir_point=[90.288, 1.531546, 220.11969, 11.69270039, 48.78041985],
            ideal_point=[8.31600002, 0.073306, 7.78305, 0.50138474, 2.05918711],
        )


class C10MOP6(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(6),
            nadir_point=[90.288, 1.531546, 220.11969, 10.5269, 2.22623795, 27.61348617],
            ideal_point=[
                8.31600001,
                0.073306,
                7.78305,
                1.67954,
                0.34696451,
                0.97636492,
            ],
        )


class C10MOP7(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(7),
            nadir_point=[
                90.288,
                1.531546,
                220.11969,
                11.69270039,
                48.78041985,
                10.5269,
                2.22623795,
                27.61348617,
            ],
            ideal_point=[
                8.31600002,
                0.073306,
                7.78305,
                0.50138474,
                2.05918711,
                1.67954,
                0.34696451,
                0.97636492,
            ],
        )


class C10MOP8(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(8),
            nadir_point=[2.60839215e-01, 1.55274600e06],
            ideal_point=[4.85086515e-02, 3.92426000e05],
        )


class C10MOP9(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=c10mop(9),
            nadir_point=[2.71840124e-01, 1.48977000e06, 2.46631744e08],
            ideal_point=[4.92413696e-02, 4.13290000e05, 8.20435840e07],
        )


class IN1KMOP1(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(1),
            nadir_point=[2.8064380e-01, 3.9453704e07],
            ideal_point=[1.63165165e-01, 6.53928000e06],
        )


class IN1KMOP2(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(2),
            nadir_point=[2.79503105e-01, 1.15401719e10],
            ideal_point=[1.62489554e-01, 6.33771744e08],
        )


class IN1KMOP3(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(3),
            nadir_point=[2.80709580e-01, 3.86586880e07, 1.26246722e10],
            ideal_point=[1.60220631e-01, 6.53928000e06, 6.33771744e08],
        )


class IN1KMOP4(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(4),
            nadir_point=[1.83413892e-01, 7.24852720e07],
            ideal_point=[1.76191352e-01, 4.16878960e07],
        )


class IN1KMOP5(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(5),
            nadir_point=[1.83367384e-01, 1.48670599e10],
            ideal_point=[1.76416579e-01, 8.84410415e09],
        )


class IN1KMOP6(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(6),
            nadir_point=[1.83166683e-01, 7.09963360e07, 1.47570987e10],
            ideal_point=[1.76592023e-01, 4.16878960e07, 8.84410415e09],
        )


class IN1KMOP7(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(7),
            nadir_point=[2.64243178e-01, 9.97696000e06],
            ideal_point=[1.68749369e-01, 4.60245600e06],
        )


class IN1KMOP8(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(8),
            nadir_point=[2.65494391e-01, 1.00460160e07, 1.33675395e09],
            ideal_point=[1.67638303e-01, 4.60245600e06, 2.06589072e08],
        )


class IN1KMOP9(MONASProblem):
    def __init__(self):
        super().__init__(
            benchmark=in1kmop(9),
            nadir_point=[2.64910135e-01, 1.02660400e07, 1.30716522e09, 6.30334821e01],
            ideal_point=[1.65323007e-01, 4.60245600e06, 2.06589072e08, 9.89034651e00],
        )


index_to_op_str = {
    0: "nor_conv_3x3",
    1: "avg_pool_3x3",
    2: "nor_conv_1x1",
    3: "skip_connect",
    4: "none",
}


class NASBench201Test(BaseProblem):
    def __init__(self, n_obj=3, nadir_point=None, ideal_point=None):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_dim=6,
            n_obj=n_obj,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )

        # from evoxbench.database.init import config
        # mo_nas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2bo_bench", "problem", "mo_nas")
        # data_path = os.path.join(mo_nas_path, "data")
        # database_path = os.path.join(mo_nas_path, "datapath")
        # config(data_path=data_path, database_path=database_path)

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
