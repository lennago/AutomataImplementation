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
    N2: int,
    alphabet_size: int,
    seed: int | str,
    epsilon: Fraction,
    delta: Fraction,
    exact: int,
    algo_res: int,
    ratio: float,
    max_size: float,
    algo: str,
    time_sec: float,
    system: str = "My Laptop",
    db_path="results.db",
):
    # normalize to Decimal for log; store string for exactness
    if exact == 0:
        return
    time_sec = max(time_sec, 0.0000001)
    ex = Decimal(str(exact))
    if algo_res < 2**23:
        ap = Decimal(float(algo_res))
    else:
        ap = Decimal(int(algo_res))

    row = dict(
        M=M,
        M2=M2,
        N=N,
        N2=N2,
        alphabet_size=alphabet_size,
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
        system=str(system),
    )
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO runs
        (M, M2, N, N2, alphabet_size, seed, epsilon, delta, exact, algo_res, ratio, max_size, time_sec, algo, log2_exact, log2_algo_res, system)
        VALUES (:M,:M2,:N,:N2,:alphabet_size,:seed,:epsilon,:delta,:exact,:algo_res,:ratio,:max_size,:time_sec,:algo,:log2_exact,:log2_algo_res,:system)
    """,
        row,
    )
    con.commit()
    con.close()
