import sqlite3

DDL = """
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
  algo      TEXT    NOT NULL,

  log2_exact  REAL  NOT NULL,
  log2_algo_res REAL  NOT NULL,
  system    TEXT    NOT NULL DEFAULT 'My Laptop'
);

CREATE INDEX IF NOT EXISTS idx_runs_algo_n   ON runs(algo, N);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at);
"""

con = sqlite3.connect("results_laptop.db")
con.executescript(DDL)
con.commit()
con.close()
print("Initialized results_laptop.db")
