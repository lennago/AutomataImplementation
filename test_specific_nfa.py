from fractions import Fraction
from main import run_main_helper
from nfa import NFA
from specific_nfas import Triple_0x90, DEADBEEF, DEADBEEF_OR_FEEDFACE, BOUNDEDGAP_4BITS

EPSILON = Fraction(9, 10)
EPSILON_MAIN = Fraction(9, 10)
DELTA = Fraction(9, 10)
DELTA_MAIN = Fraction(9, 10)

NFA = [Triple_0x90(), DEADBEEF(), DEADBEEF_OR_FEEDFACE(), BOUNDEDGAP_4BITS()]
N = 256
DEBUG = "Minimal"  # Possible debug settings: ["None", "Full", "Minimal"]


if __name__ == "__main__":
    while True:
        for nfa in NFA:
            run_main_helper(
                m=nfa.m,
                n=N,
                epsilon=EPSILON,
                epsilon_main=EPSILON_MAIN,
                delta=DELTA,
                delta_main=DELTA_MAIN,
                seed=nfa.name,
                provided_nfa=nfa.to_tuple(),
                debug=DEBUG,
            )
