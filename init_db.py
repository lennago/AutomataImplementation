import sqlite3
from log_results import log_run

DDL = """
CREATE TABLE IF NOT EXISTS algorithms (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  name      TEXT    UNIQUE NOT NULL
);


CREATE TABLE IF NOT EXISTS runs (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  M         INTEGER NOT NULL,
  M2        INTEGER NOT NULL,
  N         INTEGER NOT NULL,
  Seed      TEXT NOT NULL,
  EPSILON   REAL NOT NULL,
  DELTA     REAL NOT NULL,

  exact     TEXT    NOT NULL,   -- arbitrary precision kept as text
  algo_res  TEXT    NOT NULL,
  ratio     REAL    NOT NULL,   -- algo_res/exact

  max_size  REAL,
  time_sec  REAL    NOT NULL,
  algo      TEXT    NOT NULL REFERENCES algorithms(name) ON DELETE CASCADE,

  log2_exact  REAL  NOT NULL,
  log2_algo_res REAL  NOT NULL
);

INSERT OR IGNORE INTO algorithms (name) VALUES ('DependentFPRAS'), ('MainFPRAS'), ('BruteForce');

CREATE INDEX IF NOT EXISTS idx_runs_algo_n   ON runs(algo, N);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at);
"""

con = sqlite3.connect("results.db")
con.executescript(DDL)
con.commit()
con.close()
print("Initialized results.db")

if __name__ == "__main__":
    M = [10, 10, 10, 10, 10, 10, 10, 10, 10]
    M2 = [12, 9, 10, 12, 12, 12, 10, 12, 12]
    N = [40, 30, 60, 30, 100, 100, 100, 100, 100]
    seed = [
        8574797079797019200,
        11529889007780650590,
        9390503890170877325,
        14257273446675518432,
        15720230020624689711,
        7678108053683293329,
        14298252076838986926,
        1444566841902866431,
        7321773262983886676,
    ]
    exact = [
        733003493227,
        19036505,
        111722478689785952,
        395968315,
        52628025810003975763379879936,
        950737950171172051122527404032,
        5620755704255431607796629504,
        452732346868114145162114891776,
        542903915138928661019754496,
    ]
    algo_res = [
        731955347917,
        19013728,
        111601695219202340,
        396028759,
        52605047523611840014285129030,
        951566896937543425116583951246,
        5620305628570306848031140897,
        452330506519758958484769223351,
        542445541677301267492402809,
    ]
    ratio = [float(f"{a/e:.6f}") for e, a in zip(exact, algo_res)]
    max_size = [0.27, 0.09, 0.34, 0.2, 0.8, 0.75, 0.55, 0.77, 0.78]
    time_sec = [
        2235.80643340023,
        584.3183304999839,
        4775.876035999469,
        1088.059195400041,
        20131.2989663,
        23045.2118836,
        13627.476786400002,
        21770.7089937,
        18448.631119100006,
    ]

    epsilon = [0.1] * 9
    delta = [0.9] * 9

    algo = ["DependentFPRAS"] * 9

    for i, _ in enumerate(M):
        log_run(
            M=M[i],
            M2=M2[i],
            N=N[i],
            seed=seed[i],
            epsilon=epsilon[i],
            delta=delta[i],
            exact=exact[i],
            algo_res=algo_res[i],
            ratio=ratio[i],
            max_size=max_size[i],
            time_sec=time_sec[i],
            algo=algo[i],
        )
