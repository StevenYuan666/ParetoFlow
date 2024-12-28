import os

import numpy as np

from offline_moo.off_moo_bench.problem.base import BaseProblem

from .lambo.tasks.regex import RegexTask as InnerRegexTask

regex_task_instance_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "experiments",
    "test",
    "regex_problem.pkl",
)
rfp_task_instance_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "experiments",
    "test",
    "proxy_rfp_problem.pkl",
)
zinc_task_instance_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "experiments",
    "test",
    "zinc_problem.pkl",
)

print(regex_task_instance_file)
assert os.path.exists(regex_task_instance_file)
assert os.path.exists(rfp_task_instance_file)
assert os.path.exists(zinc_task_instance_file)

import pickle

with open(regex_task_instance_file, "rb+") as f:
    regex_task_instance = pickle.load(f)
with open(rfp_task_instance_file, "rb+") as f:
    rfp_task_instance = pickle.load(f)
with open(zinc_task_instance_file, "rb+") as f:
    zinc_task_instance = pickle.load(f)


def class2dict(data):
    dicdadta = {}
    for name in dir(data):
        value = getattr(data, name)
        if (not name.startswith("__")) and (not callable(value)):
            dicdadta[name] = value
    return dicdadta


# assert 0, class2dict(zinc_task_instance)


class REGEX(BaseProblem):
    def __init__(
        self,
    ):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=regex_task_instance.n_obj,
            n_dim=regex_task_instance.n_var,
            xl=regex_task_instance.xl,
            xu=regex_task_instance.xu,
        )
        self.task_instance = regex_task_instance
        print(type(self.task_instance))
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([0.64954899, 0.7886475, 0.73789501])

    def get_ideal_point(self):
        return np.array([-4.0, -3.79092841, -4.0])


class RFP(BaseProblem):
    def __init__(
        self,
    ):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=rfp_task_instance.n_obj,
            n_dim=rfp_task_instance.n_var,
            xl=rfp_task_instance.xl,
            xu=rfp_task_instance.xu,
        )
        self.task_instance = rfp_task_instance
        print(type(self.task_instance))
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([4.0, 4.0])

    def get_ideal_point(self):
        return np.array([-4.0, -1.36930666])


class ZINC(BaseProblem):
    def __init__(
        self,
    ):
        super().__init__(
            name=self.__class__.__name__,
            problem_type="discrete",
            n_obj=zinc_task_instance.n_obj,
            n_dim=zinc_task_instance.n_var,
            xl=zinc_task_instance.xl,
            xu=zinc_task_instance.xu,
        )
        self.task_instance = zinc_task_instance
        print(type(self.task_instance))
        self.__dict__.update(self.task_instance.__dict__)

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.task_instance.evaluate(X)

    def get_nadir_point(self):
        return np.array([1.36227612, 2.25588286])

    def get_ideal_point(self):
        return np.array([-2.17846752, -2.77324161])


# assert 0, class2dict(ZINC())
