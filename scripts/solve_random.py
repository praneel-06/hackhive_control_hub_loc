"""Read-only: solve random plans end to end, report crashes and failed checks."""
import collections, json, pickle, random, sys, time, traceback
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from hubring.plan import RESPLAN_DIR, build_plan
from hubring.solve import solve_plan

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TAG = sys.argv[2] if len(sys.argv) > 2 else None   # optional: dump per-plan rows to out/solve_random_<N>_<TAG>.json
raws = pickle.load(open(RESPLAN_DIR / "ResPlan.pkl", "rb"))
aug = set(json.load(open(RESPLAN_DIR / "split.json"))["augmented"])
random.seed(11)
idx = random.sample(range(len(raws)), N)
crash, fails, times, unsolv, ok, Hs, J, X = collections.Counter(), collections.Counter(), [], 0, 0, [], [], []
rows = []
fb = collections.Counter()
ex = collections.defaultdict(list)
for i in idx:
    pid = raws[i]["id"]
    t = time.time()
    try:
        s = solve_plan(raws[i])
    except Exception as e:
        k = f"{type(e).__name__}: {str(e)[:70]}"
        rows.append({"id": pid, "status": "crash", "error": k})
        crash[k] += 1; ex[k].append(pid)
        if len(ex[k]) == 1:
            tb = traceback.extract_tb(e.__traceback__)[-1]; ex[k].append(f"{tb.filename.split('/')[-1]}:{tb.lineno}")
        continue
    times.append(time.time() - t)
    infeasible = getattr(s, "status", "infeasible" if s.H == 0 else "") == "infeasible"
    row = {"id": pid, "status": "unsolvable" if infeasible else "solved", "s": times[-1],
           "fallback": getattr(s, "fallback", 0), "log": getattr(s, "log", [])}
    rows.append(row)
    fb[row["fallback"]] += 1
    if infeasible:
        unsolv += 1; continue
    if s.H == 0:
        fails["no cover found"] += 1; ex["no cover found"].append(pid); continue
    row.update(H=s.H, J=s.total.J, X=s.total.X, lb=s.lower_bound,
               failed=[k for k, v in s.checks.items() if v is not True])
    bad = [k for k, v in s.checks.items() if v is not True]
    for k in bad:
        fails[k] += 1; ex[k].append(pid)
    if not bad:
        ok += 1; Hs.append(s.H); J.append(s.total.J); X.append(s.total.X)
feas = N - unsolv - sum(crash.values())
print(f"{N} random plans: crashed {sum(crash.values())}, unsolvable (room>100 m² or inside wall) {unsolv}, "
      f"solved {feas}, valid {ok} ({ok/max(feas,1)*100:.1f}% of solved)")
print("crashes:", {k: v for k, v in crash.items()}, {k: ex[k][:4] for k in crash})
print("failed checks:", dict(fails), {k: ex[k][:5] for k in fails})
print(f"time/plan median {np.median(times):.2f}s p95 {np.percentile(times,95):.2f}s max {max(times):.2f}s; "
      f"mean hubs {np.mean(Hs):.2f}; median J {np.median(J):.1f}")
print(f"crossings: total {sum(X)}, plans with X>0 {sum(x > 0 for x in X)}")
print("fallback levels:", dict(fb), "; valid but flagged (clearance < 0.5 m):",
      sum(r["status"] == "solved" and not r.get("failed") and any("clearance only" in m for m in r["log"]) for r in rows))
if TAG:
    json.dump(rows, open(f"out/solve_random_{N}_{TAG}.json", "w"), indent=0, default=float)
