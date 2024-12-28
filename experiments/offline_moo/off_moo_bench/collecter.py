import os
import sys

base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(base_path)

import numpy as np
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival, calc_crowding_distance
from pymoo.core.evaluator import Evaluator
from pymoo.core.repair import Repair
from pymoo.factory import get_crossover
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling, PermutationRandomSampling
from pymoo.util.randomized_argsort import randomized_argsort

from offline_moo.off_moo_bench.problem import get_problem
from offline_moo.off_moo_bench.problem.comb_opt.mo_portfolio import PortfolioRepair
from offline_moo.off_moo_bench.problem.lambo.lambo.optimizers.mutation import (
    LocalMutation,
)
from offline_moo.off_moo_bench.problem.lambo.lambo.utils import ResidueTokenizer

load_evoxbench = True
try:
    from offline_moo.off_moo_bench.problem.mo_nas import get_genetic_operator
except:
    load_evoxbench = False


class StartFromZeroRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])
        pop.set("X", X)
        return pop


class RoundingRepair(Repair):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        return np.around(X).astype(int)


class IntegerRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        X = super()._do(problem, n_samples, **kwargs)
        return np.around(X).astype(int)


evox_sampling, evox_crossover, evox_mutation, evox_repair = (
    get_genetic_operator() if load_evoxbench else None,
    None,
    None,
    None,
)

CROSSOVERS = {
    "lambo_sbx": get_crossover(name="int_sbx", prob=0.0, eta=16),
    "evox_crossover": evox_crossover,
    "order": OrderCrossover(),
}

MUTATIONS = {
    "lambo_local": LocalMutation(
        prob=1.0, eta=16, safe_mut=False, tokenizer=ResidueTokenizer()
    ),
    "evox_mutation": evox_mutation,
    "inversion": InversionMutation(),
}

REPAIRS = {
    "start_from_zero": StartFromZeroRepair(),
    "evox_repair": evox_repair,
    "portfolio": PortfolioRepair(),
    "rounding": RoundingRepair(),
}

SAMPLINGS = {
    "evox_sampling": evox_sampling,
    "int_rnd": IntegerRandomSampling(),
    "perm_rnd": PermutationRandomSampling(),
}


def get_operator_dict(config: dict) -> dict:
    operator_dict = {}
    if "crossover" in config.keys():
        assert (
            config["crossover"] in CROSSOVERS.keys()
        ), "Crossover {crossover} not found".format(crossover=config["crossover"])
        operator_dict["crossover"] = CROSSOVERS[config["crossover"]]

    if "mutation" in config.keys():
        assert (
            config["mutation"] in MUTATIONS.keys()
        ), "Mutation {mutation} not found".format(mutation=config["mutation"])
        operator_dict["mutation"] = MUTATIONS[config["mutation"]]

    if "repair" in config.keys():
        assert config["repair"] in REPAIRS.keys(), "Repair {repair} not found".format(
            repair=config["repair"]
        )
        operator_dict["repair"] = REPAIRS[config["repair"]]

    if "sampling" in config.keys():
        assert (
            config["sampling"] in SAMPLINGS.keys()
        ), "Sampling {sampling} not found".format(sampling=config["sampling"])
        operator_dict["sampling"] = SAMPLINGS[config["sampling"]]

    return operator_dict


class AmateurRankAndCrowdSurvival(RankAndCrowdingSurvival):
    def __init__(self, p=0.6, nds=None) -> None:
        super().__init__(nds)
        self.p = p

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # if np.random.rand() < self.p:
        #     F = pop.get("F").astype(float, copy=False)
        #     survivors = np.random.randint(low=0, high=len(pop), size=n_survive)
        #     fronts = self.nds.do(F)

        #     for k, front in enumerate(fronts):
        #         # calculate the crowding distance of the front
        #         crowding_of_front = calc_crowding_distance(F[front, :])

        #         # save rank and crowding in the individual class
        #         for j, i in enumerate(front):
        #             pop[i].set("rank", k)
        #             pop[i].set("crowding", crowding_of_front[j])

        #     return pop[survivors]
        if np.random.rand() < self.p:
            F = pop.get("F").astype(float, copy=False)
            fronts = self.nds.do(F)
            survivors = []

            for k in range(len(fronts) - 1, -1, -1):
                front = fronts[k]
                # calculate the crowding distance of the front
                crowding_of_front = calc_crowding_distance(F[front, :])

                # save rank and crowding in the individual class
                for j, i in enumerate(front):
                    pop[i].set("rank", k)
                    pop[i].set("crowding", crowding_of_front[j])

                # current front sorted by crowding distance if splitting
                if len(survivors) + len(front) > n_survive:
                    I = randomized_argsort(
                        crowding_of_front, order="descending", method="numpy"
                    )
                    I = I[: (n_survive - len(survivors))]

                # otherwise take the whole front unsorted
                else:
                    I = np.arange(len(front))

                # extend the survivors by all or selected individuals
                survivors.extend(front[I])
            return pop[survivors]

        else:
            return super()._do(problem, pop, *args, n_survive=n_survive, **kwargs)


last_pop = None
last_algo = None


def molecule_callback(algorithm):
    global last_pop, last_algo
    last_algo = algorithm
    last_pop = algorithm.pop


def gen_new_individual_from_last_pop(problem, **kwargs):
    global last_pop
    if last_pop is None:
        return np.random.rand(1, problem.n_var)

    from pymoo.algorithms.moo.nsga2 import binary_tournament
    from pymoo.core.duplicate import DefaultDuplicateElimination
    from pymoo.core.mating import Mating
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.selection.tournament import TournamentSelection

    mating = Mating(
        selection=TournamentSelection(func_comp=binary_tournament),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=DefaultDuplicateElimination(),
    )

    off = mating.do(problem, last_pop, n_offsprings=1, algorithm=last_algo, **kwargs)
    return off.get("X")


class MoleculeEvaluator(Evaluator):
    def _eval(self, problem, pop, evaluate_values_of, **kwargs):
        X = pop.get("X")
        x_all = []
        res = {}

        for i, x_i in enumerate(X):
            done = False
            while not done:
                try:
                    out = problem.evaluate(
                        x_i.reshape(1, -1),
                        return_values_of=evaluate_values_of,
                        return_as_dictionary=True,
                        mode="collect",
                        **kwargs
                    )
                    for key, val in out.items():
                        if val is not None:
                            if key not in res.keys():
                                res[key] = val
                            else:
                                res[key] = np.concatenate([res[key], val], axis=0)
                    done = True
                    break
                except:
                    off_pop = gen_new_individual_from_last_pop(problem=problem)
                    if isinstance(off_pop, np.ndarray):
                        x_i = off_pop
                        continue
                    assert 0, off_pop.get("X")

            x_all.append(x_i.reshape(1, -1))

        X = np.concatenate(x_all, axis=0)

        pop.set("X", X)
        for key, val in res.items():
            if val is not None:
                pop.set(key, val)

        pop.apply(lambda ind: ind.evaluated.update(res.keys()))
