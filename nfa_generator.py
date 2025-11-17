# nfa_randgen_edges.py
# Non-layered random NFA generators targeting a desired acceptance fraction at length n.
# Output: list of (from, symbol, to), plus start state and accepting set.
# Includes exact evaluator for small n, and a "uniform over counts" sampler.
# No external deps.

from __future__ import annotations
import os
import json
from typing import List, Tuple, Set, Optional, Dict
import math
import random
from nfa import NFA, DAG
from main import BruteForcePowerset_wrapper


Transition = Tuple[int, int, int]  # (u, a, v) with a in {0,1}

# ---------- Mean-field predictor and calibration ----------


def _predict_fraction(
    m: int, n: int, p: float, alpha: float, alphabet_size: int
) -> float:
    assert alphabet_size >= 2
    if p <= 0.0:
        k = 1.0
    else:
        k = 1.0
        p_eff = 1.0 - (1.0 - p) ** alphabet_size
        log1m_p = math.log1p(-p_eff)  # < 0
        for _ in range(n):
            k = m * (1.0 - math.exp(k * log1m_p))
    return 1.0 - math.exp(k * math.log1p(-alpha))


def calibrate_p(
    m: int,
    n: int,
    r: float,
    alphabet_size: int,
    alpha: float = 0.5,
    p_lo: Optional[float] = None,
    p_hi: Optional[float] = None,
    tol: float = 1e-3,
    max_iter: int = 60,
) -> float:
    assert 0 < m and 0 <= n
    assert 0.0 < alpha < 1.0
    assert 0.0 <= r <= 1.0
    if p_lo is None:
        p_lo = 1e-6
    if p_hi is None:
        p_hi = min(5.0 / (m * alphabet_size), 0.5)
    r_lo = _predict_fraction(m, n, p_lo, alpha, alphabet_size)
    r_hi = _predict_fraction(m, n, p_hi, alpha, alphabet_size)
    if r <= r_lo:
        return p_lo
    if r >= r_hi:
        return p_hi
    lo, hi = p_lo, p_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = _predict_fraction(m, n, mid, alpha, alphabet_size)
        if abs(r_mid - r) <= tol:
            return mid
        if r_mid < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_edges(
    m: int,
    p: float,
    alpha: float,
    rng: random.Random,
    alphabet_size: int,
    start: int = 0,
    accept_count: Optional[int] = None,
) -> Tuple[List[Transition], int, Set[int]]:
    trans: List[Transition] = []
    for u in range(m):
        for v in range(m):
            for symbol in range(alphabet_size):
                if rng.random() < p:
                    trans.append((u, symbol, v))
    if accept_count is not None:
        accept_count = max(1, min(m, int(accept_count)))
        accepts = set(rng.sample(range(m), accept_count))
    else:
        accepts = {s for s in range(m) if rng.random() < alpha}
        if not accepts:
            accepts.add(rng.randrange(m))
    return trans, start, accepts


def generate_nfa(
    m: int,
    n: int,
    alphabet_size: int = 2,
    seed: Optional[int] = None,
    logging: bool = False,
) -> Tuple[List[Transition], int, Set[int], Dict]:
    """
    Choose an integer target k uniformly from {0,1,...,alphabet_size^n}, then attempt to generate
    an NFA whose EXACT number of accepted words of length n equals k.
    Returns (transitions, start, accepts, info) with 'target_count' and 'exact_count'.
    """
    start = 0
    tolerance = 0.001
    max_tries = 25
    alpha = 0.5
    rng = random.Random(seed)
    total = alphabet_size**n
    k_target = 1 + rng.randrange(total)  # Not interested in empty languages
    r_target = k_target / total if total > 0 else 0.0
    p = calibrate_p(m, n, r_target, alpha=alpha, alphabet_size=alphabet_size)
    pmin, pmax = 1e-6, min(5.0 / (m * alphabet_size), 0.5)
    history: List[Tuple[float, float]] = []
    best_pack = None
    best_gap = float("inf")
    best_r = None
    best_p = p

    for t in range(max_tries):
        transitions, s, accepts = sample_edges(
            m,
            p,
            alpha,
            rng,
            alphabet_size=alphabet_size,
            start=start,
            accept_count=None,
        )
        data = {"transitions": transitions, "start": [s], "accepts": list(accepts)}
        with open("current.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        nfa = NFA(
            m, transitions, [s], list(accepts), alphabet_size=alphabet_size, debug=False
        )
        nfa.minimize()
        nfa.reduce_to_boolean_alphabet()
        dag = DAG(nfa, n)
        r_exact = (
            BruteForcePowerset_wrapper(
                dag=dag,
                n=n,
                M=m,
                alphabet_size=alphabet_size,
                seed=f"Gen {t}: {seed}",
                printing=False,
                logging=logging,
                progress_bar=False,
            )
            / total
        )
        os.remove("current.json")
        k_exact = int(round(r_exact * total))
        gap = abs(k_exact - k_target)
        if gap < best_gap:
            best_gap = gap
            best_pack = (transitions, s, accepts)
            best_r = r_exact
            best_p = p
            if abs(r_exact - r_target) <= tolerance:
                info = {
                    "mode": "uniform_over_counts",
                    "p": p,
                    "alpha": alpha,
                    "n": n,
                    "alphabet_size": alphabet_size,
                    "target_count": k_target,
                    "exact_count": k_exact,
                    "tries": t + 1,
                    "tolerance": tolerance,
                    "seed": seed,
                    "exact_used": True,
                }
                return transitions, s, list(accepts), info
        history.append((p, r_exact))
        if len(history) >= 2:
            (p0, r0), (p1, r1) = history[-2], history[-1]
            denom = r1 - r0
            p_new = (
                p1 + (r_target - r1) * (p1 - p0) / denom
                if abs(denom) > 1e-9
                else p1 * (1.2 if r1 < r_target else 0.8)
            )
        else:
            p_new = p * (1.2 if r_exact < r_target else 0.8)
        p = max(pmin, min(pmax, p_new))

    transitions, s, accepts = best_pack
    info = {
        "mode": "uniform_over_counts",
        "p": best_p,
        "alpha": alpha,
        "n": n,
        "alphabet_size": alphabet_size,
        "target_count": k_target,
        "exact_count": int(round(best_r * total)) if best_r is not None else None,
        "tries": max_tries,
        "tolerance": tolerance,
        "seed": seed,
        "exact_used": True,
        "note": "Returned closest match after max_tries.",
    }
    return transitions, s, list(accepts), info
