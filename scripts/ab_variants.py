"""A/B solver variants on the fixed random sample, in parallel. Each variant sets
module attributes before solving; rows are written in solve_random's format.

Usage: .venv/bin/python scripts/ab_variants.py N name 'grid.WALL_OFFSET_M=0.05' 'loops.KEEP_TOP=None' ...
       → out/solve_random_N_<name>.json (compare with scripts/compare_runs.py)
Timings are measured per plan while 8 workers share the CPU: compare J, not time.
"""
import importlib
import json
import multiprocessing as mp
import pickle
import random
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.plan import RESPLAN_DIR  # noqa: E402


def _init(settings):
    warnings.filterwarnings("ignore")
    ns = {m: importlib.import_module(f"hubring.{m}") for m in ("grid", "loops", "solve", "plan", "routing")}
    for s in settings:
        target, value = s.split("=", 1)
        mod, attr = target.rsplit(".", 1)
        setattr(ns[mod], attr, eval(value, ns))


def _work(raw):
    from hubring.solve import solve_plan
    t = time.perf_counter()
    try:
        s = solve_plan(raw)
    except Exception as e:
        return {"id": raw["id"], "status": "crash", "error": f"{type(e).__name__}: {e}"}
    row = {"id": raw["id"], "s": time.perf_counter() - t,
           "status": "unsolvable" if s.status == "infeasible" else "solved", "fallback": s.fallback, "log": s.log}
    if s.H:
        row.update(H=s.H, J=s.total.J, X=s.total.X, lb=s.lower_bound,
                   failed=[k for k, v in s.checks.items() if v is not True])
    return row


if __name__ == "__main__":
    n, name, settings = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    raws = pickle.load(open(RESPLAN_DIR / "ResPlan.pkl", "rb"))
    random.seed(11)
    sub = [raws[i] for i in random.sample(range(len(raws)), n)]
    with mp.get_context("spawn").Pool(mp.cpu_count(), initializer=_init, initargs=(settings,)) as pool:
        rows = pool.map(_work, sub, chunksize=4)
    json.dump(rows, open(f"out/solve_random_{n}_{name}.json", "w"), default=float)
    print(name, settings, "→", f"out/solve_random_{n}_{name}.json")
