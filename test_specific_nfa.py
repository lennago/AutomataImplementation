from main import run_main_helper
from nfa import NFA, DAG
from fractions import Fraction
from nfa_generator import exact_fraction_edges
from specific_nfas import Triple_0x90, DEADBEEF, DEADBEEF_OR_FEEDFACE, BOUNDEDGAP_4BITS

EPSILON = Fraction(9, 10)
EPSILON_MAIN = Fraction(9, 10)
DELTA = Fraction(9, 10)

NFA = BOUNDEDGAP_4BITS()
N = 256
DEBUG = "Minimal"  # Possible debug settings: ["None", "Full", "Minimal"]


if __name__ == "__main__":
    run_main_helper(
        m=NFA.m,
        n=N,
        epsilon=EPSILON,
        epsilon_main=EPSILON_MAIN,
        delta=DELTA,
        seed=NFA.name,
        provided_nfa=NFA.to_tuple(),
        debug=DEBUG,
    )
