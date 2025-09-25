import os
from timeit import timeit
from fractions import Fraction
from typing import List, Tuple, Optional
import time
from algorithms import (
    BruteForce,
    MainFPRAS,
    DependentFPRAS,
)
from nfa import NFA, DAG
from log_results import log_run


def main(
    states: int,
    transitions: List[Tuple[int, int, int]],
    start_states: List[int],
    accept_states: List[int],
    n: int,
    M: int,
    exact: int,
    seed: int,
    epsilon_main: Fraction = Fraction(9, 10),
    epsilon: Fraction = Fraction(1, 10),
    delta: Fraction = Fraction(1, 4),
    test_time: bool = False,
    debug: str = "None",
    progress_bar: bool = True,
    RUN_MAIN: bool = False,
    RUN_BRUTEFORCE: bool = False,
):
    nfa = NFA(
        num_states=states,
        transitions=transitions,
        start_states=start_states,
        accept_states=accept_states,
    )
    seperator = "\n" * 3
    if not nfa.num_states:
        print("Language of NFA is empty.")
        return
    if not (0 < epsilon < 1):
        raise ValueError("Epsilon must be in the range (0, 1).")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in the range (0, 1).")

    nfa.minimize()
    dag = DAG(nfa, n)
    print(f"DAG has {dag.m} states per layer and {dag.n} layers.")
    print(f"DAG is {'sparse' if dag.sparse else 'dense'}.")
    if debug:
        for layer in range(dag.states.shape[0]):
            print(f"Layer {layer}: {dag.states[layer].nonzero()[0]}")
        print()
        for m in range(dag.m):
            for symbol in range(2):
                print(
                    f"Transitions from state {m} via {symbol}: {dag.transition_matrices[symbol][m].nonzero()[0]}"
                )
    print(f"\nStart time: {time.strftime('%H:%M:%S', time.localtime())}\n")
    if test_time:
        RUNS_BruteForce = 1000
        RUNS_DEPENDENT_FPRAS = 10
        time_BruteForce = timeit(
            lambda: BruteForce_wrapper(
                dag,
                n,
                M=M,
                exact=exact,
                seed=seed,
                printing=False,
                debug=False,
                progress_bar=False,
            ),
            number=RUNS_BruteForce,
        )
        print(
            f"Time taken by BruteForce Parallel for {RUNS_BruteForce} runs: {time_BruteForce:.4f} seconds\n{time_BruteForce / RUNS_BruteForce:.4f} seconds per run{seperator}"
        )
        time_dependant = timeit(
            lambda: dependentFPRAS_wrapper(
                dag,
                n,
                M,
                epsilon,
                delta,
                exact=exact,
                seed=seed,
                printing=False,
                debug="None",
                progress_bar=False,
            ),
            number=RUNS_DEPENDENT_FPRAS,
        )
        print(
            f"Time taken by Dependent FPRAS RAM for {RUNS_DEPENDENT_FPRAS} runs: {time_dependant:.4f} seconds\n{time_dependant / RUNS_DEPENDENT_FPRAS:.4f} seconds per run{seperator}"
        )
    else:
        print(
            f"Time taken by Dependent FPRAS: {timeit(lambda: dependentFPRAS_wrapper(dag, n, M, epsilon, delta, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
        )
        if RUN_BRUTEFORCE:
            print(
                f"Time taken by BruteForce Parallel: {timeit(lambda: BruteForce_wrapper(dag, n, M, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar),number=1)} seconds{seperator}"
            )
        if RUN_MAIN:
            print(
                f"Time taken by Main FPRAS: {timeit(lambda: mainFPRAS_wrapper(dag, n, M, epsilon_main, delta, exact=exact, seed=seed, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
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
            f"BruteForce algorithm result: {res} (Factor from exact: {float(res / exact):.6f}, should be 1.000000)"
        )


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
            f"Main FPRAS algorithm result: {res} (Factor from exact: {float(res / exact):.6f})"
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
            f"Dependent FPRAS algorithm result: {int(res)} (Factor from exact: {float(res / exact):.6f})"
        )


if __name__ == "__main__":
    # Example usage
    import random
    from nfa_generator import (
        tune_and_sample_edges,
        sample_edges_uniform_over_counts,
        exact_fraction_edges,
    )

    NUMBER_OF_STATES = 10
    NUMBER_OF_LAYERS = 10
    EPSILON = Fraction(1, 10)
    EPSILON_MAIN = Fraction(9, 10)
    DELTA = Fraction(9, 10)
    TARGET_COUNT = None
    SEED = None
    # Possible debug settings: ["None", "Full", "Minimal"]
    DEBUG = "Minimal"
    PROGRESS_BAR = True
    TEST_TIME = False
    RUN_MAIN = False
    RUN_BRUTEFORCE = True
    while True:
        random_data = os.urandom(8)
        seed = int.from_bytes(random_data, byteorder="big") if SEED is None else SEED
        NUMBER_OF_STATES = random.randint(2, 30)
        NUMBER_OF_LAYERS = random.randint(2, 50)
        if NUMBER_OF_LAYERS * NUMBER_OF_STATES < 30:
            RUN_MAIN = True
        else:
            RUN_MAIN = False
        if NUMBER_OF_LAYERS <= 30:
            RUN_BRUTEFORCE = True
        else:
            RUN_BRUTEFORCE = False
        trans, start, accepting, info = (
            sample_edges_uniform_over_counts(
                NUMBER_OF_STATES, NUMBER_OF_LAYERS, seed=seed
            )
            if TARGET_COUNT is None
            else tune_and_sample_edges(
                NUMBER_OF_STATES, NUMBER_OF_LAYERS, TARGET_COUNT, seed=seed
            )
        )
        print("NFA generation info:")
        for key in info.keys():
            print(f"{key}: {info[key]}")
        from_count = int(
            round(
                exact_fraction_edges(
                    NUMBER_OF_STATES, NUMBER_OF_LAYERS, trans, accepting, start
                )
                * (1 << NUMBER_OF_LAYERS)
            )
        )
        print(f"\nExact result: {from_count}\n")
        time.sleep(2)
        main(
            states=NUMBER_OF_STATES,
            transitions=trans,
            start_states=[start],
            accept_states=accepting,
            n=NUMBER_OF_LAYERS,
            M=NUMBER_OF_STATES,
            exact=from_count,
            epsilon_main=EPSILON_MAIN,
            epsilon=EPSILON,
            delta=DELTA,
            test_time=TEST_TIME,
            seed=seed,
            debug=DEBUG,
            progress_bar=PROGRESS_BAR,
            RUN_MAIN=RUN_MAIN,
            RUN_BRUTEFORCE=RUN_BRUTEFORCE,
        )
