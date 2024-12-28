try:
    # Config EvoXBench
    import os

    from evoxbench.database.init import config

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "problem", "mo_nas"
    )
    config(os.path.join(base_path, "database"), os.path.join(base_path, "data"))
except:
    pass

import numpy as np

from offline_moo.off_moo_bench.registration import make, register, registry, spec

try:
    register(
        "RE21-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE21Dataset",
        "off_moo_bench.problem.synthetic_func:RE21",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE22-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE22Dataset",
        "off_moo_bench.problem.synthetic_func:RE22",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE23-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE23Dataset",
        "off_moo_bench.problem.synthetic_func:RE23",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE24-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE24Dataset",
        "off_moo_bench.problem.synthetic_func:RE24",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE25-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE25Dataset",
        "off_moo_bench.problem.synthetic_func:RE25",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE31-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE31Dataset",
        "off_moo_bench.problem.synthetic_func:RE31",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE32-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE32Dataset",
        "off_moo_bench.problem.synthetic_func:RE32",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE33-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE33Dataset",
        "off_moo_bench.problem.synthetic_func:RE33",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE34-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE34Dataset",
        "off_moo_bench.problem.synthetic_func:RE34",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE35-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE35Dataset",
        "off_moo_bench.problem.synthetic_func:RE35",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE36-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE36Dataset",
        "off_moo_bench.problem.synthetic_func:RE36",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE37-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE37Dataset",
        "off_moo_bench.problem.synthetic_func:RE37",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE41-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE41Dataset",
        "off_moo_bench.problem.synthetic_func:RE41",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE42-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE42Dataset",
        "off_moo_bench.problem.synthetic_func:RE42",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RE61-Exact-v0",
        "off_moo_bench.datasets.continuous.re_suite_dataset:RE61Dataset",
        "off_moo_bench.problem.synthetic_func:RE61",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "VLMOP1-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:VLMOP1Dataset",
        "off_moo_bench.problem.synthetic_func:VLMOP1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "VLMOP2-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:VLMOP2Dataset",
        "off_moo_bench.problem.synthetic_func:VLMOP2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "VLMOP3-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:VLMOP3Dataset",
        "off_moo_bench.problem.synthetic_func:VLMOP3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZDT1-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT1Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZDT2-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT2Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZDT3-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT3Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZDT4-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT4Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZDT6-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:ZDT6Dataset",
        "off_moo_bench.problem.synthetic_func:ZDT6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "OmniTest-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:OmniTestDataset",
        "off_moo_bench.problem.synthetic_func:OmniTest",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ1-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ1Dataset",
        "off_moo_bench.problem.dtlz:DTLZ1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ2-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ2Dataset",
        "off_moo_bench.problem.dtlz:DTLZ2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ3-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ3Dataset",
        "off_moo_bench.problem.dtlz:DTLZ3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ4-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ4Dataset",
        "off_moo_bench.problem.dtlz:DTLZ4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ5-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ5Dataset",
        "off_moo_bench.problem.dtlz:DTLZ5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ6-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ6Dataset",
        "off_moo_bench.problem.dtlz:DTLZ6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "DTLZ7-Exact-v0",
        "off_moo_bench.datasets.continuous.synthetic_function_dataset:DTLZ7Dataset",
        "off_moo_bench.problem.dtlz:DTLZ7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "MOHopperV2-Exact-v0",
        "off_moo_bench.datasets.continuous.mo_hopper_dataset:MOHopperV2Dataset",
        "off_moo_bench.problem.morl.morl_problem:MOHopperV2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "MOSwimmerV2-Exact-v0",
        "off_moo_bench.datasets.continuous.mo_swimmer_dataset:MOSwimmerV2Dataset",
        "off_moo_bench.problem.morl.morl_problem:MOSwimmerV2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "Portfolio-Exact-v0",
        "off_moo_bench.datasets.continuous.portfolio_dataset:PortfolioDataset",
        "off_moo_bench.problem.comb_opt.mo_portfolio:MOPortfolio",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "Molecule-Exact-v0",
        "off_moo_bench.datasets.continuous.molecule_dataset:MoleculeDataset",
        "off_moo_bench.problem.moo_molecule_funcs.molecule:Molecule",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "NASBench201Test-Exact-v0",
        "off_moo_bench.datasets.discrete.nb201_test_dataset:NB201TestDataset",
        "off_moo_bench.problem.mo_nas.mo_nas:NASBench201Test",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP1-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP1Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP2-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP2Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP3-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP3Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP4-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP4Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP5-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP5Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP6-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP6Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP7-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP7Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP8-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP8Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP8",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "C10MOP9-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:C10MOP9Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:C10MOP9",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP1-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP1Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP1",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP2-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP2Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP2",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP3-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP3Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP3",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP4-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP4Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP4",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP5-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP5Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP5",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP6-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP6Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP6",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP7-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP7Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP7",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP8-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP8Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP8",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "IN1KMOP9-Exact-v0",
        "off_moo_bench.datasets.sequence.monas_dataset:IN1KMOP9Dataset",
        "off_moo_bench.problem.mo_nas.mo_nas:IN1KMOP9",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "Regex-Exact-v0",
        "off_moo_bench.datasets.sequence.regex_dataset:RegexDataset",
        "off_moo_bench.problem.lambo.lambo_mole_task:REGEX",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "ZINC-Exact-v0",
        "off_moo_bench.datasets.sequence.zinc_dataset:ZINCDataset",
        "off_moo_bench.problem.lambo.lambo_mole_task:ZINC",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "RFP-Exact-v0",
        "off_moo_bench.datasets.sequence.rfp_dataset:RFPDataset",
        "off_moo_bench.problem.lambo.lambo_mole_task:RFP",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiTSP20-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:BiTSP20Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp:BiTSP20",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiTSP50-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:BiTSP50Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp:BiTSP50",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiTSP100-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:BiTSP100Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp:BiTSP100",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiTSP500-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:BiTSP500Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp:BiTSP500",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "TriTSP20-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:TriTSP20Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp_3obj:TriTSP20",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "TriTSP50-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:TriTSP50Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp_3obj:TriTSP50",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "TriTSP100-Exact-v0",
        "off_moo_bench.datasets.permutation.motsp_dataset:TriTSP100Dataset",
        "off_moo_bench.problem.comb_opt.mo_tsp_3obj:TriTSP100",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiKP50-Exact-v0",
        "off_moo_bench.datasets.permutation.mokp_dataset:BiKP50Dataset",
        "off_moo_bench.problem.comb_opt.mo_kp:BiKP50",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiKP100-Exact-v0",
        "off_moo_bench.datasets.permutation.mokp_dataset:BiKP100Dataset",
        "off_moo_bench.problem.comb_opt.mo_kp:BiKP100",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiKP200-Exact-v0",
        "off_moo_bench.datasets.permutation.mokp_dataset:BiKP200Dataset",
        "off_moo_bench.problem.comb_opt.mo_kp:BiKP200",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiCVRP20-Exact-v0",
        "off_moo_bench.datasets.permutation.mocvrp_dataset:BiCVRP20Dataset",
        "off_moo_bench.problem.comb_opt.mo_cvrp:BiCVRP20",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiCVRP50-Exact-v0",
        "off_moo_bench.datasets.permutation.mocvrp_dataset:BiCVRP50Dataset",
        "off_moo_bench.problem.comb_opt.mo_cvrp:BiCVRP50",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass


try:
    register(
        "BiCVRP100-Exact-v0",
        "off_moo_bench.datasets.permutation.mocvrp_dataset:BiCVRP100Dataset",
        "off_moo_bench.problem.comb_opt.mo_cvrp:BiCVRP100",
        # keyword arguments for building the dataset
        dataset_kwargs=dict(max_samples=None, max_percentile=100, min_percentile=0),
        # keyword arguments for building the exact oracle
        problem_kwargs=dict(),
    )
except:
    pass
