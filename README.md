## AutomataImplementation

Algorithms and tooling to count |L^n(NFA)| for a given nondeterministic finite automaton (NFA) and word length n. The project implements and evaluates:

- MainFPRAS (from Arenas et al.)
- DependentFPRAS (memory-optimized sampling-based estimator)
- Baselines: BruteForce and BruteForcePowerset (exact via powerset dynamic programming)

Core data structures are the NFA and its induced layered DAG (both in `nfa.py`). The primary entrypoint to run and evaluate algorithms is `run_main_helper` from `main.py`.


## Quick start (Windows, PowerShell)

1) Create and activate a virtual environment

```pwsh
# From the project root
py -3.11 -m venv .venv   # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: If PowerShell blocks activation, start a new elevated PowerShell and run:

```pwsh
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

2) (Optional, recommended) Initialize result logging databases

```pwsh
python init_db.py
```

This creates `results.db` and `results_combined.db` used by `log_results.py`.

3) Run an experiment via `run_main_helper`

Launch an interactive Python and execute:

```python
from fractions import Fraction
from main import run_main_helper

run_main_helper(
		m=(2, 200),            # number of NFA states (fixed int or (min,max) for random)
		n=(2, 50),             # word length (fixed int or (min,max) for random)
		epsilon=Fraction(9, 10),
		epsilon_main=Fraction(9, 10),
		delta=Fraction(9, 10),
		delta_main=Fraction(9, 10),
		seed=None,             # optional int, or str if you also provide a custom NFA
		provided_nfa=None,     # see “Provide your own NFA” below
		logging=True           # set True to log into results_combined.db
)
```

`run_main_helper` will:
- Generate or accept a provided NFA, then minimize it
- Build the layered DAG for length n
- Compute the exact result with BruteForcePowerset (and log it if logging=True)
- Always run DependentFPRAS; optionally run MainFPRAS and BruteForce depending on small-instance limits
- Print timings and approximation factors, and log each run when logging=True


## Project structure

- `algorithms.py` — Implementations of MainFPRAS, DependentFPRAS, BruteForce, BruteForcePowerset
- `nfa.py` — `NFA` and `DAG` classes, minimization, simulation helpers
- `main.py` — Convenience wrappers and `run_main_helper` entrypoint for experiments
- `nfa_generator.py` — Random NFA generator used by `run_main_helper`
- `init_db.py` — Creates/initializes SQLite databases for logging results
- `log_results.py` — Utility to log runs into `results_combined.db`
- `merge_databases.py` — Helper to merge older result databases into the combined DB
- `nfa_algorithms_analysis.ipynb`, `plots.ipynb` — Notebooks for analyzing logged runs and plotting
- `requirements.txt` — Python dependencies for this project


## Provide your own NFA

`run_main_helper` can run on a user-specified NFA. Pass `seed` as a string and set `provided_nfa` to:

- `m`: int — number of states (without the additional start/accept consolidation)
- `transitions`: list of `(from_state, symbol, to_state)` with `symbol in {0,1}`
- `starts`: list of start state indices
- `accepts`: list of accepting state indices

In code this is typed as:

```python
TRANSITION = tuple[int, int, int]  # (from_state, symbol, to_state)
NFA_TYPE = tuple[int, list[TRANSITION], list[int], list[int]]
```

Example:

```python
from fractions import Fraction
from main import run_main_helper

m = 4
transitions = [
		(0, 0, 1), (0, 1, 2),
		(1, 0, 3), (2, 1, 3),
]
starts = [0]
accepts = [3]

run_main_helper(
		m=m,
		n=6,
		epsilon=Fraction(9,10), epsilon_main=Fraction(9,10),
		delta=Fraction(9,10),   delta_main=Fraction(9,10),
		seed="custom",                 # string -> required when providing an NFA
		provided_nfa=(m, transitions, starts, accepts),
		logging=True
)
```

Important: If `seed` is a string, `provided_nfa` must be set and `provided_nfa[0] == m`.


## Controls and options

`run_main_helper(...)` parameters (selected):

- `m`, `n`: int or tuple[min,max]. If a tuple is given, a random value in the range is chosen per run.
- `epsilon`, `delta`, `epsilon_main`, `delta_main`: `fractions.Fraction` in (0,1).
- `debug`: "None" | "Minimal" | "Full". Prints internal details; "Full" can be very verbose.
- `progress_bar`: Show tqdm progress bars.
- `logging`: When True, logs each algorithm run into `results_combined.db` (initialize with `init_db.py` first).
- `run_main_limit` (default 40): Only run MainFPRAS if `m*n < run_main_limit`.
- `run_bruteforce_limit` (default 30): Only run BruteForce if `n <= run_bruteforce_limit`.

You can also run `python main.py` directly. The `__main__` section loops indefinitely with default ranges; press Ctrl+C to stop. Prefer calling `run_main_helper` yourself for deterministic, single runs.


## Algorithms (from `algorithms.py`)

- `MainFPRAS(dag, n, epsilon, delta, debug="None", progress_bar=True).run() -> float`
	- Implements the FPRAS from Arenas et al. Assumes a single start and accept state in the DAG (enforced by `NFA` construction and minimization).

- `DependentFPRAS(dag, n, epsilon, delta, debug="None", progress_bar=True).run() -> tuple[float,int]`
	- Memory-optimized estimator that reports `(estimate, max_memory_gb)`; the wrappers convert to friendly output and handle logging.

- `BruteForce(dag, n).run() -> int`
	- Enumerates all words of length n in parallel. Only feasible for small n.

- `BruteForcePowerset(dag, n).run() -> tuple[int, float]`
	- Exact dynamic programming over the powerset of reachable state sets. Returns `(exact_count, max_memory_gb)`.

Wrapper functions in `main.py` (used by `run_main_helper`) handle timing and logging: `mainFPRAS_wrapper`, `dependentFPRAS_wrapper`, `BruteForce_wrapper`, `BruteForcePowerset_wrapper`.


## Results and analysis

- Results are stored in `results_combined.db` (table `runs`) with columns: sizes (M, M2, N), parameters (epsilon, delta, seed), algorithm, timings, ratio, log2 values, etc. See `init_db.py` and `log_results.py`.
- Use the notebooks `nfa_algorithms_analysis.ipynb` and `plots.ipynb` to explore and plot results.
- `merge_databases.py` can import and normalize older DB files into `results_combined.db`.


## Notes and tips

- Python 3.10+ is recommended. Recent NumPy/SciPy versions are pinned in `requirements.txt` and have prebuilt wheels for Windows.
- On large instances, the exact `BruteForcePowerset` is still preferred over brute force; the sampling algorithms are designed for much larger sizes but may use multiple CPU cores and significant RAM.
- To reproduce a run, pin a fixed integer `seed`; when generating random NFAs, `run_main_helper` derives a seed if `seed=None`.


## License

MIT — see `LICENSE`.

