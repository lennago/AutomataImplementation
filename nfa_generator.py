# nfa_randgen_edges.py
# Non-layered random NFA generators targeting a desired acceptance fraction at length n.
# Output: list of (from, symbol, to), plus start state and accepting set.
# Includes exact evaluator for small n, and a "uniform over counts" sampler.
# No external deps.

from __future__ import annotations
from typing import List, Tuple, Set, Optional, Dict
import math, random

Transition = Tuple[int, int, int]  # (u, a, v) with a in {0,1}

# ---------- Mean-field predictor and calibration ----------


def _predict_fraction(m: int, n: int, p: float, alpha: float) -> float:
    if p <= 0.0:
        k = 1.0
    else:
        k = 1.0
        log1m_p = math.log1p(-p)  # < 0
        for _ in range(n):
            k = m * (1.0 - math.exp(k * log1m_p))
    return 1.0 - math.exp(k * math.log1p(-alpha))


def calibrate_p(
    m: int,
    n: int,
    r: float,
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
        p_hi = min(5.0 / m, 0.5)
    r_lo = _predict_fraction(m, n, p_lo, alpha)
    r_hi = _predict_fraction(m, n, p_hi, alpha)
    if r <= r_lo:
        return p_lo
    if r >= r_hi:
        return p_hi
    lo, hi = p_lo, p_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = _predict_fraction(m, n, mid, alpha)
        if abs(r_mid - r) <= tol:
            return mid
        if r_mid < r:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------- Random NFA samplers (edges as (u,a,v)) ----------


def sample_edges_bernoulli(
    m: int,
    p: float,
    alpha: float,
    rng: random.Random,
    start: int = 0,
    accept_count: Optional[int] = None,
) -> Tuple[List[Transition], int, Set[int]]:
    trans: List[Transition] = []
    for u in range(m):
        for v in range(m):
            if rng.random() < p:
                trans.append((u, 0, v))
            if rng.random() < p:
                trans.append((u, 1, v))
    if accept_count is not None:
        accept_count = max(1, min(m, int(accept_count)))
        accepts = set(rng.sample(range(m), accept_count))
    else:
        accepts = {s for s in range(m) if rng.random() < alpha}
        if not accepts:
            accepts.add(rng.randrange(m))
    return trans, start, accepts


def _poisson_knuth(lam: float, rng: random.Random) -> int:
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def sample_edges_bounded(
    m: int,
    lam: float,
    alpha: float,
    rng: random.Random,
    start: int = 0,
    accept_count: Optional[int] = None,
) -> Tuple[List[Transition], int, Set[int]]:
    trans: List[Transition] = []
    for a in (0, 1):
        for u in range(m):
            deg = _poisson_knuth(lam, rng)
            deg = max(0, min(m, deg))
            if deg > 0:
                targets = rng.sample(range(m), deg)
                for v in targets:
                    trans.append((u, a, v))
    if accept_count is not None:
        accept_count = max(1, min(m, int(accept_count)))
        accepts = set(rng.sample(range(m), accept_count))
    else:
        accepts = {s for s in range(m) if rng.random() < alpha}
        if not accepts:
            accepts.add(rng.randrange(m))
    return trans, start, accepts


# ---------- Evaluators on edge list ----------


def _build_adj(m: int, transitions: List[Transition]):
    adj0 = [[] for _ in range(m)]
    adj1 = [[] for _ in range(m)]
    for u, a, v in transitions:
        if a == 0:
            adj0[u].append(v)
        else:
            adj1[u].append(v)
    return adj0, adj1


def estimate_fraction_edges(
    m: int,
    n: int,
    transitions: List[Transition],
    accepts: Set[int],
    start: int = 0,
    T: int = 5000,
    seed: Optional[int] = None,
) -> float:
    rng = random.Random(seed)
    adj0, adj1 = _build_adj(m, transitions)
    accepted = 0
    for _ in range(T):
        states = {start}
        for _t in range(n):
            if not states:
                break
            E = adj0 if rng.getrandbits(1) == 0 else adj1
            nxt = set()
            for u in states:
                if E[u]:
                    nxt.update(E[u])
            states = nxt
        else:
            if states and (accepts & states):
                accepted += 1
    return accepted / T if T > 0 else 0.0


def exact_fraction_edges(
    m: int,
    n: int,
    transitions: List[Transition],
    accepts: Set[int],
    start: Set[int] | int = {0},
) -> float:
    out0 = [0] * m
    out1 = [0] * m
    for u, a, v in transitions:
        if a == 0:
            out0[u] |= 1 << v
        else:
            out1[u] |= 1 << v
    start_mask = 0
    if isinstance(start, int):
        start_mask |= 1 << start
    else:
        for s in start:
            start_mask |= 1 << s
    accept_mask = 0
    for s in accepts:
        accept_mask |= 1 << s
    cur = {start_mask: 1}
    for _ in range(n):
        nxt = {}
        for mask, cnt in cur.items():
            m0 = 0
            mm = mask
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                m0 |= out0[u]
                mm ^= lsb
            m1 = 0
            mm = mask
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                m1 |= out1[u]
                mm ^= lsb
            nxt[m0] = nxt.get(m0, 0) + cnt
            nxt[m1] = nxt.get(m1, 0) + cnt
        cur = nxt
    total = 1 << n
    acc_words = sum(cnt for mask, cnt in cur.items() if (mask & accept_mask) != 0)
    return acc_words / total if total > 0 else 0.0


# ---------- One-shot driver ----------


def tune_and_sample_edges(
    m: int,
    n: int,
    r: float,
    *,
    alpha: float = 0.5,
    degree: str = "bernoulli",
    tolerance: float = 0.001,
    T: int = 5000,
    max_tries: int = 25,
    seed: Optional[int] = None,
    lam: Optional[float] = None,
    start: int = 0,
    accept_count: Optional[int] = None,
    exact_if_small: bool = True,
    exact_limit: int = (1 << 18),
) -> Tuple[List[Transition], int, Set[int], Dict]:
    rng = random.Random(seed)
    p = calibrate_p(m, n, r, alpha=alpha)
    pmin, pmax = 1e-6, min(5.0 / m, 0.5)
    chosen_lam = lam if lam is not None else p * m
    history: List[Tuple[float, float]] = []
    last_pack = None
    last_rstar = None
    for t in range(max_tries):
        if degree == "bernoulli":
            transitions, s, accepts = sample_edges_bernoulli(
                m, p, alpha, rng, start=start, accept_count=accept_count
            )
        elif degree == "bounded":
            chosen_lam = max(0.0, min(chosen_lam, m))
            transitions, s, accepts = sample_edges_bounded(
                m, chosen_lam, alpha, rng, start=start, accept_count=accept_count
            )
        else:
            raise ValueError("degree must be 'bernoulli' or 'bounded'")
        if exact_if_small and (1 << n) <= exact_limit:
            r_star = exact_fraction_edges(m, n, transitions, accepts, start=s)
        else:
            r_star = estimate_fraction_edges(
                m, n, transitions, accepts, start=s, T=T, seed=rng.randrange(1 << 30)
            )
        if abs(r_star - r) <= tolerance:
            info = {
                "degree": degree,
                "p": p,
                "lam": chosen_lam if degree == "bounded" else None,
                "alpha": alpha,
                "r_target": r,
                "r_est": r_star,
                "tries": t + 1,
                "T": T,
                "seed": seed,
                "exact_used": bool(exact_if_small and (1 << n) <= exact_limit),
            }
            return transitions, s, list(accepts), info
        last_pack, last_rstar = (transitions, s, accepts), r_star
        history.append((p, r_star))
        if len(history) >= 2:
            (p0, r0), (p1, r1) = history[-2], history[-1]
            denom = r1 - r0
            p_new = (
                p1 + (r - r1) * (p1 - p0) / denom
                if abs(denom) > 1e-9
                else p1 * (1.2 if r1 < r else 0.8)
            )
        else:
            p_new = p * (1.2 if r_star < r else 0.8)
        p = max(pmin, min(pmax, p_new))
        if degree == "bounded":
            chosen_lam = p * m
    transitions, s, accepts = last_pack
    info = {
        "degree": degree,
        "p": p,
        "lam": chosen_lam if degree == "bounded" else None,
        "alpha": alpha,
        "r_target": r,
        "r_est": last_rstar,
        "tries": max_tries,
        "T": T,
        "seed": seed,
        "note": "Best-effort; consider increasing T or max_tries.",
        "exact_used": bool(exact_if_small and (1 << n) <= exact_limit),
    }
    return transitions, s, list(accepts), info


# ---------- Uniform-over-counts convenience -----------------------------------


def sample_edges_uniform_over_counts(
    m: int,
    n: int,
    *,
    alpha: float = 0.5,
    degree: str = "bernoulli",
    seed: Optional[int] = None,
    accept_count: Optional[int] = None,
    lam: Optional[float] = None,
    start: int = 0,
    tolerance: float = 0.001,
    max_tries: int = 25,
) -> Tuple[List[Transition], int, Set[int], Dict]:
    """
    Choose an integer target k uniformly from {0,1,...,2^n}, then attempt to generate
    an NFA whose EXACT number of accepted words of length n equals k. Uses the exact
    evaluator (n must be small enough). Falls back to the closest match within max_tries.
    Returns (transitions, start, accepts, info) with 'target_count' and 'exact_count'.
    """
    rng = random.Random(seed)
    total = 1 << n
    k_target = 1 + rng.randrange(total)  # Not interested in empty languages
    r_target = k_target / total if total > 0 else 0.0
    p = calibrate_p(m, n, r_target, alpha=alpha)
    pmin, pmax = 1e-6, min(5.0 / m, 0.5)
    chosen_lam = lam if lam is not None else p * m
    history: List[Tuple[float, float]] = []
    best_pack = None
    best_gap = float("inf")
    best_r = None
    best_p = p

    for t in range(max_tries):
        if degree == "bernoulli":
            transitions, s, accepts = sample_edges_bernoulli(
                m, p, alpha, rng, start=start, accept_count=accept_count
            )
        elif degree == "bounded":
            chosen_lam = max(0.0, min(chosen_lam, m))
            transitions, s, accepts = sample_edges_bounded(
                m, chosen_lam, alpha, rng, start=start, accept_count=accept_count
            )
        else:
            raise ValueError("degree must be 'bernoulli' or 'bounded'")

        r_exact = exact_fraction_edges(m, n, transitions, accepts, start=s)
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
                    "degree": degree,
                    "p": p,
                    "lam": chosen_lam if degree == "bounded" else None,
                    "alpha": alpha,
                    "n": n,
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
        if degree == "bounded":
            chosen_lam = p * m

    transitions, s, accepts = best_pack
    info = {
        "mode": "uniform_over_counts",
        "degree": degree,
        "p": best_p,
        "lam": chosen_lam if degree == "bounded" else None,
        "alpha": alpha,
        "n": n,
        "target_count": k_target,
        "exact_count": int(round(best_r * total)) if best_r is not None else None,
        "tries": max_tries,
        "tolerance": tolerance,
        "seed": seed,
        "exact_used": True,
        "note": "Returned closest match after max_tries.",
    }
    return transitions, s, list(accepts), info


# ---------- Example ----------
if __name__ == "__main__":
    from nfa import NFA, DAG
    from algorithms import BruteForce

    # uniform-over-counts example at tiny m,n
    m, n = 2, 2
    trans, start, accepts, info = sample_edges_uniform_over_counts(
        m, n, alpha=0.5, degree="bernoulli"
    )
    print("uniform-over-counts info:", info)
    nfa = NFA(
        num_states=m,
        transitions=trans,
        start_states=[start],
        accept_states=list(accepts),
    )
    dag = DAG(nfa, n)
    from_count = int(
        round(exact_fraction_edges(m, n, trans, accepts, start) * (1 << n))
    )
    print("exact count:", from_count)
    print(f"BruteForceParallel result: {BruteForce(dag, n).run()}")
