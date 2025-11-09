"""import sys
import os
import math
import numpy as np
from fractions import Fraction
from tqdm.auto import tqdm
from timeit import timeit
from nfa import NFA, DAG
from algorithms import BruteForce, BruteForcePowerset
import random
from nfa_generator import (
    sample_edges_uniform_over_counts,
)
from log_results import log_run
from main import BruteForce_wrapper, BruteForcePowerset_wrapper

BruteForce_Limit = 25
M = (20, 300)
N = (2, 40)

while True:
    m = random.randint(M[0], M[1])
    n = random.randint(N[0], N[1])
    print(f"Testing m:{m}, n:{n}")
    TARGET_COUNT = None
    SEED = None
    # Possible debug settings: ["None", "Full", "Minimal"]
    DEBUG = "None"
    PROGRESS_BAR = True
    random_data = os.urandom(8)
    seed = int.from_bytes(random_data, byteorder="big") if SEED is None else SEED
    trans, start, accepting, info = sample_edges_uniform_over_counts(m, n, seed=seed)
    nfa = NFA(m, trans, [start], accepting, debug=False)
    nfa.minimize()
    dag = DAG(nfa, n)
    print(f"Resulting m:{dag.m}")
    if n <= BruteForce_Limit:
        BruteForce_wrapper(
            dag=dag,
            n=n,
            M=m,
            exact=info["exact_count"],
            seed=seed,
            printing=False,
        )"""

import json
from main import BruteForcePowerset_wrapper, dependentFPRAS_wrapper
from nfa import NFA, DAG

with open("failed_nfas.json", "r", encoding="utf-8") as f:
    failed_nfas = json.load(f)["failed-nfas"]

for nfa_data in failed_nfas:
    print(f"NFA with m={nfa_data['m']} and n={nfa_data['n']}")
    nfa = NFA(
        nfa_data["m"], nfa_data["transitions"], nfa_data["start"], nfa_data["accepts"]
    )
    nfa.minimize()
    dag = DAG(nfa, nfa_data["n"])
    print(f"Minimized to m={dag.m}")
    exact = BruteForcePowerset_wrapper(
        dag, dag.n, nfa_data["m"], seed=f"failed_nfa_{nfa_data['id']}"
    )
    dependentFPRAS_wrapper(
        dag,
        dag.n,
        nfa_data["m"],
        epsilon=0.9,
        delta=0.9,
        seed=f"failed_nfa_{nfa_data['id']}",
        exact=exact,
        debug="minimal",
    )
