# log_run.py
import sqlite3
from fractions import Fraction
from decimal import Decimal, getcontext

getcontext().prec = 300
LN2 = Decimal(2).ln()


def log2_decimal(x: Decimal) -> float:
    return float(x.ln() / LN2)


def log_run(
    M: int,
    M2: int,
    N: int,
    seed: int,
    epsilon: Fraction,
    delta: Fraction,
    exact: int,
    algo_res: int,
    ratio: float,
    max_size: float,
    algo: str,
    time_sec: float,
    db_path="results.db",
):
    # normalize to Decimal for log; store string for exactness
    ex = Decimal(str(exact))
    if algo_res < 2**23:
        ap = Decimal(float(algo_res))
    else:
        ap = Decimal(int(algo_res))

    row = dict(
        M=M,
        M2=M2,
        N=N,
        seed=str(seed),
        epsilon=float(epsilon),
        delta=float(delta),
        exact=str(exact),
        algo_res=str(ap),
        ratio=float(ratio),
        max_size=max_size,
        time_sec=float(time_sec),
        algo=str(algo),
        log2_exact=log2_decimal(ex),
        log2_algo_res=log2_decimal(ap),
    )

    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO runs
        (M, M2, N, seed, epsilon, delta, exact, algo_res, ratio, max_size, time_sec, algo, log2_exact, log2_algo_res)
        VALUES (:M,:M2,:N,:seed,:epsilon,:delta,:exact,:algo_res,:ratio,:max_size,:time_sec,:algo,:log2_exact,:log2_algo_res)
    """,
        row,
    )
    con.commit()
    con.close()


# Example usage from your experiment script:
# log_run(M, M2, N, exact_value, approx_value, ratio_value, size_bytes, elapsed_seconds, "FPRAS-v1")
