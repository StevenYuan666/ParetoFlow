import numpy as np

from offline_moo.off_moo_bench.problem.morl.base import MORLProblem


class MOHopperV2(MORLProblem):
    def __init__(self, n_obj=2, env_name="MO-Hopper-v2"):
        nadir_point = np.array([154.8246919, 1681.81775345]) * 2
        ideal_point = np.array([-1179.36239177, -1370.84501144])

        super().__init__(
            name=self.__class__.__name__,
            n_obj=n_obj,
            env_name=env_name,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )


class MOSwimmerV2(MORLProblem):
    def __init__(self, n_obj=2, env_name="MO-Swimmer-v2"):
        nadir_point = np.array([24.00957411, -25.47507349]) * 2
        ideal_point = np.array([-219.65312991, -149.99974259])

        super().__init__(
            name=self.__class__.__name__,
            n_obj=n_obj,
            env_name=env_name,
            nadir_point=nadir_point,
            ideal_point=ideal_point,
        )
