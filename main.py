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


def main(
    states: int,
    transitions: List[Tuple[int, int, int]],
    start_states: List[int],
    accept_states: List[int],
    n: int,
    epsilon_main: Fraction = Fraction(9, 10),
    epsilon: Fraction = Fraction(1, 10),
    delta: Fraction = Fraction(1, 4),
    test_time: bool = False,
    seed: Optional[int] = None,
    debug: bool = False,
    progress_bar: bool = True,
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
        exit(0)
    if not (0 < epsilon < 1):
        # raise ValueError("Epsilon must be in the range (0, 1).")
        pass
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
        RUNS_BruteForce_PARALLEL = 1000
        RUNS_DEPENDENT_FPRAS = 10
        time_BruteForce_parallel = timeit(
            lambda: BruteForce_parallel_wrapper(
                dag, n, printing=False, debug=False, progress_bar=False
            ),
            number=RUNS_BruteForce_PARALLEL,
        )
        print(
            f"Time taken by BruteForce Parallel for {RUNS_BruteForce_PARALLEL} runs: {time_BruteForce_parallel:.4f} seconds\n{time_BruteForce_parallel / RUNS_BruteForce_PARALLEL:.4f} seconds per run{seperator}"
        )
        time_dependant = timeit(
            lambda: dependentFPRAS_wrapper(
                dag,
                n,
                epsilon,
                delta,
                printing=False,
                debug=False,
                progress_bar=False,
            ),
            number=RUNS_DEPENDENT_FPRAS,
        )
        print(
            f"Time taken by Dependent FPRAS RAM for {RUNS_DEPENDENT_FPRAS} runs: {time_dependant:.4f} seconds\n{time_dependant / RUNS_DEPENDENT_FPRAS:.4f} seconds per run{seperator}"
        )
    else:
        print(
            f"Time taken by Dependent FPRAS: {timeit(lambda: dependentFPRAS_wrapper(dag, n, epsilon, delta, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
        )
        print(
            f"Time taken by BruteForce Parallel: {timeit(lambda: BruteForce_parallel_wrapper(dag, n, debug=debug, progress_bar=progress_bar),number=1)} seconds{seperator}"
        )
        print(
            f"Time taken by Main FPRAS: {timeit(lambda: mainFPRAS_wrapper(dag, n, epsilon_main, delta, debug=debug, progress_bar=progress_bar), number=1)} seconds{seperator}"
        )


def BruteForce_parallel_wrapper(
    dag: DAG,
    n: int,
    printing: bool = True,
    debug: bool = False,
    progress_bar: bool = True,
):
    """Wrapper function for the BruteForce_Parallel algorithm."""
    res = BruteForce(dag, n, debug=debug, progress_bar=progress_bar).run()
    if printing:
        print(f"BruteForce algorithm result: {res}")


def mainFPRAS_wrapper(
    dag: DAG,
    n: int,
    epsilon: Fraction,
    delta: Fraction,
    printing: bool = True,
    debug: bool = False,
    progress_bar: bool = True,
):
    """
    Main algorithm wrapper function.
    """
    res = MainFPRAS(
        dag, n, epsilon, delta, debug=debug, progress_bar=progress_bar
    ).run()
    if printing:
        print(f"Main FPRAS algorithm result: {res}")


def dependentFPRAS_wrapper(
    dag: DAG,
    n: int,
    epsilon: Fraction,
    delta: Fraction,
    printing: bool = True,
    debug: bool = False,
    progress_bar: bool = True,
):
    """
    Wrapper function for the DependentFPRAS algorithm.
    """
    res = float(
        DependentFPRAS(
            dag, n, epsilon, delta, debug=debug, progress_bar=progress_bar
        ).run()
    )
    if printing:
        print(f"Dependent FPRAS algorithm result: {res}")


if __name__ == "__main__":
    # Example usage
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
    DEBUG = True
    PROGRESS_BAR = True
    TEST_TIME = False

    random_data = os.urandom(8)
    seed = int.from_bytes(random_data, byteorder="big") if SEED is None else SEED
    trans, start, accepting, info = (
        sample_edges_uniform_over_counts(NUMBER_OF_STATES, NUMBER_OF_LAYERS, seed=seed)
        if TARGET_COUNT is None
        else tune_and_sample_edges(
            NUMBER_OF_STATES, NUMBER_OF_LAYERS, TARGET_COUNT, seed=seed
        )
    )
    print("NFA generation info:")
    for key in info.keys():
        print(f"{key}: {info[key]}")
    nfa = NFA(
        num_states=NUMBER_OF_STATES,
        transitions=trans,
        start_states=[start],
        accept_states=accepting,
    )
    dag = DAG(nfa, NUMBER_OF_LAYERS)
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
        epsilon_main=EPSILON_MAIN,
        epsilon=EPSILON,
        delta=DELTA,
        test_time=TEST_TIME,
        seed=seed,
        debug=DEBUG,
        progress_bar=PROGRESS_BAR,
    )
