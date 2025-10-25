from decimal import Decimal
import os
import random
from timeit import timeit
from fractions import Fraction
from typing import List, Tuple, Optional
import time
from algorithms import (
    BruteForce,
    MainFPRAS,
    DependentFPRAS,
    BruteForcePowerset,
)
from nfa import NFA, DAG
from log_results import log_run, log2_decimal

TRANSITION = Tuple[int, int, int]  # (from_state, symbol, to_state)
NFA_TYPE = Tuple[
    int, List[TRANSITION], List[int], List[int]
]  # (m, transitions, starts, accepts)


def main(
    dag: DAG,
    n: int,
    M: int,
    seed: int | str,
    epsilon_main: Fraction = Fraction(9, 10),
    epsilon: Fraction = Fraction(1, 10),
    delta: Fraction = Fraction(9, 10),
    delta_main: Fraction = Fraction(9, 10),
    debug: str = "None",
    progress_bar: bool = True,
    run_main: bool = False,
    run_bruteforce: bool = False,
):
    seperator = "\n" * 3
    print(f"\nStart time: {time.strftime('%H:%M:%S', time.localtime())}\n")
    exact = BruteForcePowerset_wrapper(
        dag=dag,
        n=n,
        M=M,
        seed=seed,
        printing=True,
        debug=debug,
        progress_bar=progress_bar,
    )
    print(
        f"Time taken by Dependent FPRAS: {timeit(lambda: dependentFPRAS_wrapper(dag, n, M, epsilon, delta, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
    )
    if run_bruteforce:
        print(
            f"Time taken by BruteForce Parallel: {timeit(lambda: BruteForce_wrapper(dag, n, M, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar),number=1)} seconds{seperator}"
        )
    if run_main:
        print(
            f"Time taken by Main FPRAS: {timeit(lambda: mainFPRAS_wrapper(dag, n, M, epsilon_main, delta_main, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
        )


def BruteForce_wrapper(
    dag: DAG,
    n: int,
    M: int,
    exact: int,
    seed: int,
    printing: bool = True,
    debug: str = "None",
    progress_bar: bool = True,
):
    """Wrapper function for the BruteForce_Parallel algorithm."""
    start = time.time()
    res = BruteForce(dag, n, debug=debug, progress_bar=progress_bar).run()
    log_run(
        M=M,
        M2=dag.m,
        N=n,
        seed=seed,
        epsilon=Fraction(0, 1),
        delta=Fraction(0, 1),
        exact=exact,
        algo_res=res,
        ratio=float(res / exact) if exact != 0 else float(res + 1 / 1),
        max_size=0.0,
        time_sec=time.time() - start,
        algo="BruteForce",
    )
    if printing:
        print(
            f"BruteForce algorithm result: {res} (Factor from exact: {(float(res / exact) if exact != 0 else float(res + 1 / 1)):.6f}, should be 1.000000)"
        )


def BruteForcePowerset_wrapper(
    dag: DAG,
    n: int,
    M: int,
    seed: int,
    printing: bool = True,
    debug: str = "None",
    progress_bar: bool = True,
):
    """Wrapper function for the BruteForcePowerset algorithm."""
    start = time.time()
    res, size = BruteForcePowerset(dag, n, debug=debug, progress_bar=progress_bar).run()
    log_run(
        M=M,
        M2=dag.m,
        N=n,
        seed=seed,
        epsilon=Fraction(0, 1),
        delta=Fraction(0, 1),
        exact=res,
        algo_res=res,
        ratio=1.0,
        max_size=size,
        time_sec=time.time() - start,
        algo="BruteForcePowerset",
    )
    if printing:
        print(f"\nExact result: {res}, log_2:{log2_decimal(Decimal(res))}\n")
    return res


def mainFPRAS_wrapper(
    dag: DAG,
    n: int,
    M: int,
    epsilon: Fraction,
    delta: Fraction,
    exact: int,
    seed: int,
    printing: bool = True,
    debug: str = "None",
    progress_bar: bool = True,
):
    """
    Main algorithm wrapper function.
    """
    start = time.time()
    res = MainFPRAS(
        dag, n, epsilon, delta, debug=debug, progress_bar=progress_bar
    ).run()
    log_run(
        M=M,
        M2=dag.m,
        N=n,
        seed=seed,
        epsilon=epsilon,
        delta=delta,
        exact=exact,
        algo_res=res,
        ratio=float(res / exact) if exact != 0 else float(res + 1 / 1),
        max_size=0.0,
        time_sec=time.time() - start,
        algo="MainFPRAS",
    )
    if printing:
        print(f"Correct result: {exact}")
        print(
            f"Main FPRAS algorithm result: {res} (Factor from exact: {(float(res / exact) if exact != 0 else float(res + 1 / 1)):.6f})"
        )


def dependentFPRAS_wrapper(
    dag: DAG,
    n: int,
    M: int,
    epsilon: Fraction,
    delta: Fraction,
    exact: int,
    seed: int,
    printing: bool = True,
    debug: str = "None",
    progress_bar: bool = True,
):
    """
    Wrapper function for the DependentFPRAS algorithm.
    """
    start = time.time()
    res, max_size = DependentFPRAS(
        dag, n, epsilon, delta, debug=debug, progress_bar=progress_bar
    ).run()
    log_run(
        M=M,
        M2=dag.m,
        N=n,
        seed=seed,
        epsilon=epsilon,
        delta=delta,
        exact=exact,
        algo_res=res,
        ratio=float(res / exact) if exact != 0 else float(res + 1 / 1),
        max_size=max_size,
        time_sec=time.time() - start,
        algo="DependentFPRAS",
    )
    if printing:
        print(f"Correct result: {exact}")
        print(
            f"Dependent FPRAS algorithm result: {int(res)} (Factor from exact: {(float(res / exact) if exact != 0 else float(res + 1 / 1)):.6f})"
        )


def run_main_helper(
    m: int | Tuple[int, int],
    n: int | Tuple[int, int],
    epsilon: Fraction | Tuple[Fraction, Fraction],
    epsilon_main: Fraction | Tuple[Fraction, Fraction],
    delta: Fraction | Tuple[Fraction, Fraction],
    delta_main: Fraction | Tuple[Fraction, Fraction],
    seed: Optional[int | str] = None,
    provided_nfa: Optional[NFA_TYPE] = None,
    run_main_limit: int = 40,
    run_bruteforce_limit: int = 30,
    target_count: Optional[int] = None,
    debug: str = "None",
    progress_bar: bool = True,
):
    from nfa_generator import (
        tune_and_sample_edges,
        sample_edges_uniform_over_counts,
    )

    if isinstance(n, tuple):
        n = random.randint(n[0], n[1])
    if isinstance(epsilon, tuple):
        epsilon = Fraction(random.uniform(float(epsilon[0]), float(epsilon[1])))
    if isinstance(epsilon_main, tuple):
        epsilon_main = Fraction(
            random.uniform(float(epsilon_main[0]), float(epsilon_main[1]))
        )
    if isinstance(delta, tuple):
        delta = Fraction(random.uniform(float(delta[0]), float(delta[1])))
    if isinstance(delta_main, tuple):
        delta_main = Fraction(
            random.uniform(float(delta_main[0]), float(delta_main[1]))
        )

    if not (0 < epsilon < 1):
        raise ValueError("Epsilon must be in the range (0, 1).")
    if not (0 < epsilon_main < 1):
        raise ValueError("Epsilon_main must be in the range (0, 1).")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in the range (0, 1).")
    if not (0 < delta_main < 1):
        raise ValueError("Delta_main must be in the range (0, 1).")
    if isinstance(seed, str):
        if provided_nfa is None:
            print("If seed is a string, NFA must be provided.")
            return
        if provided_nfa[0] != m:
            print("Provided NFA does not match the provided m.")
            return
        m, transitions, starts, accepts = provided_nfa
        nfa = NFA(
            m, transitions, starts, accepts, debug=False if debug == "None" else True
        )
        trans = []
        for symbol, trans_mat in enumerate(nfa.transition_matrices):
            if nfa.sparse:
                trans_mat = trans_mat.todense()
            for from_state, to_state in zip(*trans_mat.nonzero()):
                trans.append((int(from_state), symbol, int(to_state)))
    else:
        if seed is None:
            random_data = os.urandom(8)
            seed = int.from_bytes(random_data, byteorder="big")
        if isinstance(m, tuple):
            m = random.randint(m[0], m[1])
        trans, start, accepting, info = (
            sample_edges_uniform_over_counts(m, n, seed=seed)
            if target_count is None
            else tune_and_sample_edges(m, n, target_count, seed=seed)
        )
        nfa = NFA(
            m,
            trans,
            [start],
            accepting,
            debug=False if debug == "None" else True,
        )
        print("NFA generation info:")
        for key in info.keys():
            print(f"{key}: {info[key]}")  # Give user time to read exact result
    if not nfa.num_states:
        print("Language of NFA is empty.")
        return
    nfa.minimize()
    dag = DAG(nfa, n)
    if dag.m * n < run_main_limit:
        run_main = True
    else:
        run_main = False
    if n <= run_bruteforce_limit:
        run_bruteforce = True
    else:
        run_bruteforce = False
    if debug is not None:
        print(f"DAG has {dag.m} states per layer and {dag.n} layers.")
        print(f"DAG is {'sparse' if dag.sparse else 'dense'}.")
        for layer in range(dag.states.shape[0]):
            print(f"Layer {layer}: {dag.states[layer].nonzero()[0]}")
        print()
        for q in range(dag.m):
            for symbol in range(2):
                print(
                    f"Transitions from state {q} via {symbol}: {dag.transition_matrices[symbol][q].nonzero()[0]}"
                )
        print(
            f"Epsilon: {float(epsilon):.6f}, Epsilon Main: {float(epsilon_main):.6f}, Delta: {float(delta):.6f}"
        )
    main(
        dag=dag,
        n=n,
        M=m,
        seed=seed,
        epsilon_main=epsilon_main,
        epsilon=epsilon,
        delta=delta,
        delta_main=delta_main,
        debug=debug,
        progress_bar=progress_bar,
        run_main=run_main,
        run_bruteforce=run_bruteforce,
    )


if __name__ == "__main__":
    # Configuration
    # Either set fixed values (int or Fraction) or tuple (min, max) for random generation
    NUMBER_OF_STATES = (2, 200)
    NUMBER_OF_LAYERS = (2, 50)
    EPSILON = Fraction(9, 10)
    EPSILON_MAIN = Fraction(9, 10)
    DELTA = Fraction(9, 10)
    DELTA_MAIN = Fraction(9, 10)
    TARGET_COUNT = None
    SEED = None
    # Possible debug settings: ["None", "Full", "Minimal"]
    DEBUG = "Minimal"
    PROGRESS_BAR = True
    TEST_TIME = False
    RUN_MAIN_LIMIT = 40
    RUN_BRUTEFORCE_LIMIT = 25
    while True:
        run_main_helper(
            m=NUMBER_OF_STATES,
            n=NUMBER_OF_LAYERS,
            epsilon=EPSILON,
            epsilon_main=EPSILON_MAIN,
            delta=DELTA,
            delta_main=DELTA_MAIN,
            seed=SEED,
            provided_nfa=None,
            run_main_limit=RUN_MAIN_LIMIT,
            run_bruteforce_limit=RUN_BRUTEFORCE_LIMIT,
            target_count=TARGET_COUNT,
            debug=DEBUG,
            progress_bar=PROGRESS_BAR,
        )
