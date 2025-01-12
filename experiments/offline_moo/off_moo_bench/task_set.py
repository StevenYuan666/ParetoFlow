SyntheticFunctionDict = {
    "zdt1": "ZDT1-Exact-v0",
    "zdt2": "ZDT2-Exact-v0",
    "zdt3": "ZDT3-Exact-v0",
    "zdt4": "ZDT4-Exact-v0",
    "zdt6": "ZDT6-Exact-v0",
    "omnitest": "OmniTest-Exact-v0",
    "vlmop1": "VLMOP1-Exact-v0",
    "vlmop2": "VLMOP2-Exact-v0",
    "vlmop3": "VLMOP3-Exact-v0",
    "dtlz1": "DTLZ1-Exact-v0",
    "dtlz2": "DTLZ2-Exact-v0",
    "dtlz3": "DTLZ3-Exact-v0",
    "dtlz4": "DTLZ4-Exact-v0",
    "dtlz5": "DTLZ5-Exact-v0",
    "dtlz6": "DTLZ6-Exact-v0",
    "dtlz7": "DTLZ7-Exact-v0",
}

MONASSequenceDict = {
    "c10mop1": "C10MOP1-Exact-v0",
    "c10mop2": "C10MOP2-Exact-v0",
    "c10mop3": "C10MOP3-Exact-v0",
    "c10mop4": "C10MOP4-Exact-v0",
    "c10mop5": "C10MOP5-Exact-v0",
    "c10mop6": "C10MOP6-Exact-v0",
    "c10mop7": "C10MOP7-Exact-v0",
    "c10mop8": "C10MOP8-Exact-v0",
    "c10mop9": "C10MOP9-Exact-v0",
    "in1kmop1": "IN1KMOP1-Exact-v0",
    "in1kmop2": "IN1KMOP2-Exact-v0",
    "in1kmop3": "IN1KMOP3-Exact-v0",
    "in1kmop4": "IN1KMOP4-Exact-v0",
    "in1kmop5": "IN1KMOP5-Exact-v0",
    "in1kmop6": "IN1KMOP6-Exact-v0",
    "in1kmop7": "IN1KMOP7-Exact-v0",
    "in1kmop8": "IN1KMOP8-Exact-v0",
    "in1kmop9": "IN1KMOP9-Exact-v0",
}

MONASLogitsDict = {
    "nb201_test": "NASBench201Test-Exact-v0",
}

MOCOPermutationDict = {
    "motsp_20": "BiTSP20-Exact-v0",
    "motsp_50": "BiTSP50-Exact-v0",
    "motsp_100": "BiTSP100-Exact-v0",
    "motsp_500": "BiTSP500-Exact-v0",
    "motsp3obj_20": "TriTSP20-Exact-v0",
    "motsp3obj_50": "TriTSP50-Exact-v0",
    "motsp3obj_100": "TriTSP100-Exact-v0",
    "mocvrp_20": "BiCVRP20-Exact-v0",
    "mocvrp_50": "BiCVRP50-Exact-v0",
    "mocvrp_100": "BiCVRP100-Exact-v0",
    "mokp_50": "BiKP50-Exact-v0",
    "mokp_100": "BiKP100-Exact-v0",
    "mokp_200": "BiKP200-Exact-v0",
    "bi_tsp_20": "BiTSP20-Exact-v0",
    "bi_tsp_50": "BiTSP50-Exact-v0",
    "bi_tsp_100": "BiTSP100-Exact-v0",
    "bi_tsp_500": "BiTSP500-Exact-v0",
    "tri_tsp_20": "TriTSP20-Exact-v0",
    "tri_tsp_50": "TriTSP50-Exact-v0",
    "tri_tsp_100": "TriTSP100-Exact-v0",
    "bi_cvrp_20": "BiCVRP20-Exact-v0",
    "bi_cvrp_50": "BiCVRP50-Exact-v0",
    "bi_cvrp_100": "BiCVRP100-Exact-v0",
    "bi_kp_50": "BiKP50-Exact-v0",
    "bi_kp_100": "BiKP100-Exact-v0",
    "bi_kp_200": "BiKP200-Exact-v0",
}

MOCOContinuousDict = {"portfolio": "Portfolio-Exact-v0"}

MORLDict = {
    "mo_swimmer_v2": "MOSwimmerV2-Exact-v0",
    "mo_hopper_v2": "MOHopperV2-Exact-v0",
}

ScientificDesignContinuousDict = {
    "molecule": "Molecule-Exact-v0",
}

ScientificDesignSequenceDict = {
    "regex": "Regex-Exact-v0",
    "zinc": "ZINC-Exact-v0",
    "rfp": "RFP-Exact-v0",
}


RESuiteDict = {
    "re21": "RE21-Exact-v0",
    "re22": "RE22-Exact-v0",
    "re23": "RE23-Exact-v0",
    "re24": "RE24-Exact-v0",
    "re25": "RE25-Exact-v0",
    "re31": "RE31-Exact-v0",
    "re32": "RE32-Exact-v0",
    "re33": "RE33-Exact-v0",
    "re34": "RE34-Exact-v0",
    "re35": "RE35-Exact-v0",
    "re36": "RE36-Exact-v0",
    "re37": "RE37-Exact-v0",
    "re41": "RE41-Exact-v0",
    "re42": "RE42-Exact-v0",
    "re61": "RE61-Exact-v0",
}

SyntheticFunction = list(SyntheticFunctionDict.values())
MONASSequence = list(MONASSequenceDict.values())
MONASLogits = list(MONASLogitsDict.values())
MOCOPermutation = list(MOCOPermutationDict.values())
MOCOContinuous = list(MOCOContinuousDict.values())
MORL = list(MORLDict.values())
ScientificDesignContinuous = list(ScientificDesignContinuousDict.values())
ScientificDesignSequence = list(ScientificDesignSequenceDict.values())
RESuite = list(RESuiteDict.values())

MONAS = MONASSequence + MONASLogits
MOCO = MOCOPermutation + MOCOContinuous
ScientificDesign = ScientificDesignContinuous + ScientificDesignSequence

ALLTASKS = SyntheticFunction + MONAS + MOCO + MORL + ScientificDesign + RESuite
ALLTASKSDICT = {
    **SyntheticFunctionDict,
    **MONASSequenceDict,
    **MONASLogitsDict,
    **MOCOPermutationDict,
    **MOCOContinuousDict,
    **MORLDict,
    **ScientificDesignContinuousDict,
    **ScientificDesignSequenceDict,
    **RESuiteDict,
}

CONTINUOUSTASKS = (
    SyntheticFunction
    + MONASLogits
    + MOCOContinuous
    + MORL
    + ScientificDesignContinuous
    + RESuite
)
PERMUTATIONTASKS = MOCOPermutation
SEQUENCETASKS = MONASSequence + ScientificDesignSequence
