"""Solve many ResPlan plans in parallel (all cores) and report crashes, failed
checks, fallback levels and timings.

Usage:
  .venv/bin/python scripts/solve_parallel.py all full        # whole dataset → out/parallel_full.json
  .venv/bin/python scripts/solve_parallel.py 1000 quick      # random 1,000 (seed 11)
  options: --seed S  --workers W  --timeout SECONDS (hard per-plan limit, default 60)
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import pickle
import random
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.plan import ROOT, RESPLAN_DIR       # noqa: E402
from hubring.solve import FALLBACK_LEVELS, solve_plan  # noqa: E402

TIMEOUT_S = 60


class PlanTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise PlanTimeout()


def _init(timeout):
    warnings.filterwarnings("ignore")
    signal.signal(signal.SIGALRM, _alarm)
    global _TIMEOUT
    _TIMEOUT = timeout


def _solve_one(raw: dict) -> dict:
    pid = raw["id"]
    t = time.perf_counter()
    signal.alarm(_TIMEOUT)
    try:
        s = solve_plan(raw)
    except PlanTimeout:
        return {"id": pid, "status": "timeout", "s": time.perf_counter() - t}
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return {"id": pid, "status": "crash", "s": time.perf_counter() - t,
                "error": f"{type(e).__name__}: {str(e)[:70]}",
                "where": f"{tb.filename.split('/')[-1]}:{tb.lineno}"}
    finally:
        signal.alarm(0)
    row = {"id": pid, "status": s.status, "s": time.perf_counter() - t,
           "fallback": FALLBACK_LEVELS[s.fallback], "log": s.log}
    if s.H:
        row.update(H=s.H, lb=s.lower_bound, J=s.total.J, L=s.total.L, B=s.total.B,
                   X=s.total.X, O=s.total.O,
                   failed=[k for k, v in s.checks.items() if v is not True])
    return row


def summarise(rows: list[dict]) -> str:
    st = collections.Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "valid"]
    feas = len(rows) - st["infeasible"]
    crash = collections.Counter(f'{r["error"]} @ {r["where"]}' for r in rows if r["status"] == "crash")
    fails = collections.Counter(k for r in rows for k in r.get("failed", []))
    fb = collections.Counter(r["fallback"] for r in rows if "fallback" in r and r["status"] != "infeasible")
    flagged = sum(any("clearance only" in m for m in r.get("log", [])) for r in ok)
    times = np.array([r["s"] for r in rows])
    ex = lambda pred: [r["id"] for r in rows if pred(r)][:6]
    lines = [
        f"{len(rows)} plans: valid {len(ok)} / {feas} feasible ({len(ok) / max(feas, 1) * 100:.2f}%), "
        f"infeasible {st['infeasible']}, partial {st['partial']}, crashed {st['crash']}, timeout {st['timeout']}",
        f"crashes: {dict(crash)}  e.g. {ex(lambda r: r['status'] == 'crash')}",
        f"failed checks (partial plans): {dict(fails)}  e.g. {ex(lambda r: r['status'] == 'partial')}",
        f"timeouts e.g. {ex(lambda r: r['status'] == 'timeout')}",
        f"fallback levels: {dict(fb)}",
        f"valid but flagged (clearance < 0.5 m, room too narrow): {flagged}",
    ]
    if ok:
        H = np.array([r["H"] for r in ok]); J = np.array([r["J"] for r in ok]); X = np.array([r["X"] for r in ok])
        lines += [f"hubs mean {H.mean():.3f}, at lower bound {np.mean(H == [r['lb'] for r in ok]) * 100:.1f}%; "
                  f"J median {np.median(J):.1f} mean {J.mean():.1f}; crossings total {X.sum()}, plans with X>0 {(X > 0).sum()}"]
    lines += [f"time/plan median {np.median(times):.2f}s p95 {np.percentile(times, 95):.2f}s max {times.max():.2f}s"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", default="1000")
    ap.add_argument("tag", nargs="?", default="run")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--timeout", type=int, default=TIMEOUT_S)
    a = ap.parse_args()

    raws = pickle.load(open(RESPLAN_DIR / "ResPlan.pkl", "rb"))
    if a.n == "all":
        idx = list(range(len(raws)))
    else:
        random.seed(a.seed)
        idx = random.sample(range(len(raws)), int(a.n))
    t0 = time.time()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.timeout,), maxtasksperchild=200) as pool:
        for k, row in enumerate(pool.imap_unordered(_solve_one, (raws[i] for i in idx), chunksize=4), 1):
            rows.append(row)
            if k % 1000 == 0:
                print(f"  {k}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "out" / f"parallel_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    import json
    json.dump(sorted(rows, key=lambda r: r["id"]), open(out, "w"), default=float)
    print(summarise(rows))
    print(f"wall time {time.time() - t0:.0f}s on {a.workers} workers → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
