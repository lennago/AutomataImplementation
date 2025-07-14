from algorithms import Stupid, MainFPRAS, DependentFPRAS, Stupid_Parallel
from nfa import NFA
from timeit import timeit
import numpy as np
from fractions import Fraction
import math
import multiprocessing


def main(
    states: int,
    transitions: list[tuple[int, int, int]],
    start_states: list[int],
    accept_states: list[int],
    n: int,
    epsilon: Fraction = Fraction(1, 3),
    delta: Fraction = Fraction(1 / math.exp(multiprocessing.cpu_count() / 8)),
):
    nfa = NFA(
        num_states=states,
        transitions=transitions,
        start_states=start_states,
        accept_states=accept_states,
    )
    if not nfa.num_states:
        raise ValueError("Language of NFA is empty.")
    if not (0 < epsilon < 1):
        raise ValueError("Epsilon must be in the range (0, 1).")
    if not (0 < delta < 1):
        raise ValueError("Delta must be in the range (0, 1).")
    # print(timeit(lambda: dependentFPRAS_wrapper(nfa, n, epsilon, delta), number=1))
    # print(timeit(lambda: mainFPRAS_wrapper(nfa, n, epsilon), number=1))
    print(timeit(lambda: stupid_parallel_wrapper(nfa, n), number=1))
    print(timeit(lambda: stupid_wrapper(nfa, n), number=1))
    print(timeit(lambda: dependentFPRAS_wrapper(nfa, n, epsilon, delta), number=1))


def stupid_wrapper(nfa, n):
    """
    Wrapper function for the Stupid algorithm.
    This is a placeholder for any additional logic you might want to implement.
    """
    print("Stupid algorithm result:", Stupid(nfa, n).run())


def stupid_parallel_wrapper(nfa, n):
    """Wrapper function for the Stupid_Parallel algorithm.
    This is a placeholder for any additional logic you might want to implement.
    """
    print("Stupid Parallel algorithm result:", Stupid_Parallel(nfa, n).run())


def mainFPRAS_wrapper(nfa, n, epsilon):
    """
    Main algorithm wrapper function.
    This is a placeholder for any additional logic you might want to implement.
    """
    print("Main FPRAS algorithm result:", MainFPRAS(nfa, n, epsilon).run())


def dependentFPRAS_wrapper(nfa, n, epsilon, delta):
    """
    Wrapper function for the DependentFPRAS algorithm.
    This is a placeholder for any additional logic you might want to implement.
    """
    print(
        "Dependent FPRAS algorithm result:",
        round(DependentFPRAS(nfa, n, epsilon, delta).run()),
    )


if __name__ == "__main__":
    # Example usage
    NUMBER_OF_STATES = 10
    NUMBER_OF_LAYERS = 10
    TRANSITIONS_PERCENTAGE = 0.2
    # """
    example_transitions = []

    for example_state in range(NUMBER_OF_STATES):
        for symbol in [0, 1]:
            k = np.random.binomial(NUMBER_OF_STATES, TRANSITIONS_PERCENTAGE)
            np.random.binomial(NUMBER_OF_STATES, TRANSITIONS_PERCENTAGE)
            next_states = np.random.choice(range(NUMBER_OF_STATES), k, replace=False)
            example_transitions += [
                (example_state, symbol, next_state) for next_state in next_states
            ]
    main(
        states=NUMBER_OF_STATES,
        transitions=example_transitions,
        start_states=[np.random.randint(NUMBER_OF_STATES)],
        accept_states=[np.random.randint(NUMBER_OF_STATES)],
        n=NUMBER_OF_LAYERS,
    )
