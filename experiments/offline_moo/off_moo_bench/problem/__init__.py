import warnings

from offline_moo.off_moo_bench.problem.comb_opt import *
from offline_moo.off_moo_bench.problem.dtlz import *
from offline_moo.off_moo_bench.problem.synthetic_func import *

load_nas = True
try:
    from offline_moo.off_moo_bench.problem.mo_nas import *
except:
    load_nas = False
    warnings.warn(
        "Failed to config EvoXBench module. It might fail when you are running with MO-NAS tasks."
    )

load_morl = True
try:
    from offline_moo.off_moo_bench.problem.morl import *
except:
    load_morl = False
    warnings.warn(
        "Failed to config MuJoCo module. It might fail when you are running with MORL tasks."
    )

load_lambo = True
try:
    from offline_moo.off_moo_bench.problem.lambo import *
except:
    load_lambo = False
    warnings.warn(
        "Failed to config LAMBO module. It might fail when you are running with Sci-Design tasks."
    )

load_chem = True
try:
    from offline_moo.off_moo_bench.problem.moo_molecule_funcs import *
except:
    load_chem = False
    warnings.warn(
        "Failed to config Molecule module. It might fail when you are running with Sci-Design tasks."
    )


def get_problem(env_name, *args, **kwargs):
    env_name = env_name.lower()

    PROBLEMS = {
        "vlmop1": VLMOP1,
        "vlmop2": VLMOP2,
        "vlmop3": VLMOP3,
        "kursawe": Kursawe,
        "omnitest": OmniTest,
        "sympart": SYMPARTRotated,
        "zdt1": ZDT1,
        "zdt2": ZDT2,
        "zdt3": ZDT3,
        "zdt4": ZDT4,
        "zdt6": ZDT6,
        "dtlz1": DTLZ1,
        "dtlz2": DTLZ2,
        "dtlz3": DTLZ3,
        "dtlz4": DTLZ4,
        "dtlz5": DTLZ5,
        "dtlz6": DTLZ6,
        "dtlz7": DTLZ7,
        "re21": RE21,
        "re22": RE22,
        "re23": RE23,
        "re24": RE24,
        "re25": RE25,
        "re31": RE31,
        "re32": RE32,
        "re33": RE33,
        "re34": RE34,
        "re35": RE35,
        "re36": RE36,
        "re37": RE37,
        "re41": RE41,
        "re42": RE42,
        "re61": RE61,
        "re91": RE91,
        # 'qm9': QM9,
        "motsp_500": BiTSP500,
        "motsp_100": BiTSP100,
        "motsp_50": BiTSP50,
        "motsp_20": BiTSP20,
        "mokp_200": BiKP200,
        "mokp_100": BiKP100,
        "mokp_50": BiKP50,
        "portfolio": MOPortfolio,
        "mocvrp_100": BiCVRP100,
        "mocvrp_50": BiCVRP50,
        "mocvrp_20": BiCVRP20,
        "motsp3obj_100": TriTSP100,
        "motsp3obj_50": TriTSP50,
        "motsp3obj_20": TriTSP20,
        # 'mo_nas': MO_NAS,
        #     'mo_hopper_v2': MO_Hopper_V2,
        #     'mo_swimmer_v2': MO_Swimmer_V2,
    }

    if load_nas:
        PROBLEMS["nb201_test"] = NASBench201Test

        PROBLEMS["c10mop1"] = C10MOP1
        PROBLEMS["c10mop2"] = C10MOP2
        PROBLEMS["c10mop3"] = C10MOP3
        PROBLEMS["c10mop4"] = C10MOP4
        PROBLEMS["c10mop5"] = C10MOP5
        PROBLEMS["c10mop6"] = C10MOP6
        PROBLEMS["c10mop7"] = C10MOP7
        PROBLEMS["c10mop8"] = C10MOP8
        PROBLEMS["c10mop9"] = C10MOP9

        PROBLEMS["in1kmop1"] = IN1KMOP1
        PROBLEMS["in1kmop2"] = IN1KMOP2
        PROBLEMS["in1kmop3"] = IN1KMOP3
        PROBLEMS["in1kmop4"] = IN1KMOP4
        PROBLEMS["in1kmop5"] = IN1KMOP5
        PROBLEMS["in1kmop6"] = IN1KMOP6
        PROBLEMS["in1kmop7"] = IN1KMOP7
        PROBLEMS["in1kmop8"] = IN1KMOP8
        PROBLEMS["in1kmop9"] = IN1KMOP9

    if load_morl:
        PROBLEMS["mo_hopper_v2"] = MOHopperV2
        PROBLEMS["mo_swimmer_v2"] = MOSwimmerV2

    if load_chem:
        PROBLEMS["molecule"] = Molecule

    if load_lambo:
        PROBLEMS["regex"] = REGEX
        PROBLEMS["rfp"] = RFP
        PROBLEMS["zinc"] = ZINC

    if env_name not in PROBLEMS.keys():
        raise Exception(f"Problem {env_name} not found or an importing error occurred.")

    return PROBLEMS[env_name](*args, **kwargs)
