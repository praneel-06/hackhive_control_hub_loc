"""Local web UI for the hub-ring solver.

Run:  .venv/bin/python -m uvicorn app.server:app --port 8000
Then open http://localhost:8000

The 300-plan sample is available at once; the full ResPlan.pkl (if present)
loads in the background and is indexed by plan id. Plans can also be added by
uploading a ResPlan-style .pkl (one plan dict or a list of them).
"""
from __future__ import annotations

import pickle
import random
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import multiprocessing as mp
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from shapely.geometry import MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from hubring.groups import room_adjacency                      # noqa: E402
from hubring.plan import RESPLAN_DIR, DeviceError, load_plans   # noqa: E402
from hubring.solve import FALLBACK_LEVELS, solve_plan, to_json, top_solutions  # noqa: E402

PHASE_DIR = ROOT / "out" / "ui_phases"
PHASES = [
    ("p0_input", "0 · Input plan"), ("p1_clean", "1 · Clean-up & free space"), ("p2_devices", "2 · Devices"),
    ("p3_grid", "3 · Routing grid"), ("p3_door_zoom", "3 · Door lanes (zoom)"), ("p4_graph", "4 · Room graph"),
    ("p5_zones", "5 · Candidate zones"), ("p6_leg", "6 · One leg"), ("p7_orders", "7 · Visiting orders"),
    ("p7_loop", "7 · Loop: shortest vs collision-aware"), ("p8_hub", "8 · Hub on the loop"),
    ("p9_final", "9 · Final answer"),
]

app = FastAPI(title="Cable Ring Planner")

# ---------------------------------------------------------------- plan store
PLANS: dict[int, dict] = {}
SOURCE = {"sample": 0, "full": 0, "uploaded": 0, "full_loading": False, "full_error": None}
_render_lock = threading.Lock()


def _add(raws, kind):
    for r in raws:
        PLANS[int(r["id"])] = r
    SOURCE[kind] += len(raws)


_add(load_plans(), "sample")


def _load_full():
    path = RESPLAN_DIR / "ResPlan.pkl"
    if not path.exists():
        return
    SOURCE["full_loading"] = True
    try:
        with open(path, "rb") as f:
            _add(pickle.load(f), "full")
    except Exception as e:                          # keep the UI usable on the sample
        SOURCE["full_error"] = f"{type(e).__name__}: {e}"
    SOURCE["full_loading"] = False


threading.Thread(target=_load_full, daemon=True).start()


# ---------------------------------------------------------------- geometry → JSON
def _path(geom) -> str:
    """SVG path data for a (multi)polygon in metres (holes as extra subpaths)."""
    if geom is None or geom.is_empty:
        return ""
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    out = []
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        for ring in [p.exterior, *p.interiors]:
            c = np.asarray(ring.coords)
            out.append("M" + "L".join(f"{x:.3f},{y:.3f}" for x, y in c) + "Z")
    return "".join(out)


def _names(rooms) -> list[str]:
    count = Counter(r.type for r in rooms)
    seen = Counter()
    names = []
    for r in rooms:
        seen[r.type] += 1
        names.append(f"{r.type} {seen[r.type]}" if count[r.type] > 1 else r.type)
    return names


def _nodes(body) -> dict | None:
    nodes = (body or {}).get("nodes") or {}
    return {int(k): (float(v[0]), float(v[1])) for k, v in nodes.items()} or None


def _solve(raw, nodes):
    try:
        return solve_plan(raw, nodes=nodes)
    except DeviceError as e:
        raise HTTPException(400, str(e))


def _payload(raw: dict, nodes: dict | None = None, sol=None) -> dict:
    t0 = time.perf_counter()
    sol = sol or _solve(raw, nodes)
    solve_s = sol.seconds if sol.seconds else time.perf_counter() - t0
    plan = sol.plan
    names = _names(plan.rooms)
    zone_of = {r: k for k, rg in enumerate(sol.rings) for r in rg.order}
    grid = getattr(sol, "grid", None)
    node_xy = grid.pts[grid.node_pt] if grid is not None else np.array([r.node for r in plan.rooms])
    x0, y0, x1, y1 = plan.inner.bounds
    for r in plan.rooms:
        bx0, by0, bx1, by1 = r.poly.bounds
        x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    out = {
        "id": plan.id, "status": sol.status, "fallback": FALLBACK_LEVELS[sol.fallback],
        "log": sol.log, "solve_s": solve_s, "scale": plan.scale,
        "bounds": [x0, y0, x1, y1],
        "rooms": [{"i": i, "id": r.id, "type": r.type, "name": names[i], "area": r.area,
                   "path": _path(r.poly), "zone": zone_of.get(i),
                   "node": [float(v) for v in node_xy[i]], "clearance": r.clearance}
                  for i, r in enumerate(plan.rooms)],
        "excluded": [{"id": n, "type": n.rsplit("_", 1)[0], "path": _path(g)} for n, g in plan.excluded],
        "walls": _path(plan.walls), "windows": _path(plan.windows),
        "doors": "".join(_path(d) for d in plan.doors),
        "H": sol.H, "lower_bound": getattr(sol, "lower_bound", None),
        "metrics": {"J": sol.total.J, "L": sol.total.L, "B": sol.total.B, "X": sol.total.X, "O": sol.total.O},
        "checks": sol.checks, "valid": sol.valid,
        "hubs": [{"k": k, "rooms": [names[i] for i in rg.order], "room_idx": list(rg.order),
                  "ring": np.round(rg.ring, 3).tolist(), "hub": [float(v) for v in rg.hub],
                  "method": rg.hub_method, "area": rg.group.area, "devices": len(rg.order),
                  "metrics": {"J": rg.J, "L": rg.metrics.L, "B": rg.metrics.B, "X": rg.metrics.X, "O": rg.metrics.O}}
                 for k, rg in enumerate(sol.rings)],
        "grid": None, "graph": [],
    }
    if grid is not None:
        seg = grid.pts[grid.edges]
        out["grid"] = np.round(seg.reshape(len(seg), 4), 3).tolist()
        adj = room_adjacency(plan, grid)
        out["graph"] = [[int(u), int(v), d["via"]] for u, v, d in adj.edges(data=True)]
    out["server_s"] = time.perf_counter() - t0
    return out


def _raw(pid: int) -> dict:
    if pid not in PLANS:
        msg = "still loading the full dataset, try again in a few seconds" if SOURCE["full_loading"] else "unknown plan id"
        raise HTTPException(404, f"Plan {pid}: {msg}")
    return PLANS[pid]


# ---------------------------------------------------------------- API
@app.get("/api/status")
def status():
    return {"plans": len(PLANS), **SOURCE}


@app.get("/api/plans")
def plans():
    return {"ids": sorted(PLANS)}


@app.get("/api/random")
def random_plan():
    return {"id": random.choice(list(PLANS))}


@app.get("/api/solve/{pid}")
def solve(pid: int):
    return JSONResponse(_payload(_raw(pid)))


@app.post("/api/solve/{pid}")
def solve_moved(pid: int, body: dict = Body(default={})):
    """Solve with some devices moved by the user: {"nodes": {room_index: [x, y]}} in metres."""
    return JSONResponse(_payload(_raw(pid), _nodes(body)))


@app.post("/api/export/{pid}")
def export(pid: int, body: dict = Body(default={})):
    return JSONResponse(to_json(_solve(_raw(pid), _nodes(body))))


@app.post("/api/top3/{pid}")
def top3(pid: int, body: dict = Body(default={})):
    raw, nodes = _raw(pid), _nodes(body)
    t0 = time.perf_counter()
    try:
        sols = top_solutions(raw, k=3, nodes=nodes)
    except DeviceError as e:
        raise HTTPException(400, str(e))
    for s_ in sols:
        s_.seconds = 0.0
    return {"seconds": time.perf_counter() - t0, "options": [_payload(raw, sol=s_) for s_ in sols]}


# ---------------------------------------------------------------- batch jobs
JOBS: dict[str, dict] = {}
_pool: ProcessPoolExecutor | None = None


def _get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        from solve_parallel import _init
        _pool = ProcessPoolExecutor(mp.cpu_count(), mp_context=mp.get_context("spawn"),
                                    initializer=_init, initargs=(60,))
    return _pool


def _summary(rows: list[dict]) -> dict:
    st = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "valid"]
    feas = len(rows) - st["infeasible"]
    t = np.array([r["s"] for r in rows]) if rows else np.zeros(1)
    out = {"done": len(rows), "status": dict(st), "valid_pct": 100 * len(ok) / feas if feas else None,
           "time_median": float(np.median(t)), "time_p95": float(np.percentile(t, 95)), "time_max": float(t.max())}
    if ok:
        H = np.array([r["H"] for r in ok]); J = np.array([r["J"] for r in ok]); X = np.array([r["X"] for r in ok])
        out.update(hubs_mean=float(H.mean()), at_lower_bound_pct=float(100 * np.mean(H == [r["lb"] for r in ok])),
                   J_median=float(np.median(J)), J_mean=float(J.mean()), J_p90=float(np.percentile(J, 90)),
                   X_total=int(X.sum()), X_plans=int((X > 0).sum()),
                   J_hist=np.histogram(J, bins=20)[0].tolist(), J_edges=np.histogram(J, bins=20)[1].tolist())
    return out


def _run_job(job: dict, raws: list[dict]):
    from solve_parallel import _solve_one
    pool = _get_pool()
    t0 = time.perf_counter()
    futs = [pool.submit(_solve_one, r) for r in raws]
    for f in as_completed(futs):
        try:
            row = f.result()
        except Exception as e:
            row = {"id": -1, "status": "crash", "s": 0.0, "error": str(e)}
        row.pop("log", None)
        job["rows"].append(row)
        job["elapsed"] = time.perf_counter() - t0
    job["running"] = False


@app.post("/api/batch")
def batch(body: dict = Body(default={})):
    n, seed = body.get("n", 100), int(body.get("seed", 11))
    pool_ids = sorted(PLANS)
    if n == "all":
        ids = pool_ids
    else:
        n = max(1, min(int(n), len(pool_ids)))
        ids = random.Random(seed).sample(pool_ids, n)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = job = {"id": job_id, "total": len(ids), "rows": [], "running": True, "elapsed": 0.0,
                          "workers": mp.cpu_count(), "seed": seed}
    threading.Thread(target=_run_job, args=(job, [PLANS[i] for i in ids]), daemon=True).start()
    return {"job": job_id, "total": len(ids)}


@app.get("/api/batch/{job_id}")
def batch_status(job_id: str, rows: bool = False):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown batch job")
    rws = list(job["rows"])
    out = {k: job[k] for k in ("id", "total", "running", "elapsed", "workers", "seed")}
    out["summary"] = _summary(rws)
    if rows:
        out["rows"] = rws
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """A ResPlan-style pickle: one plan dict or a list of them. Pickle can run
    code when loaded, so only upload files you trust (this is a local tool)."""
    data = await file.read()
    try:
        obj = pickle.loads(data)
    except Exception as e:
        raise HTTPException(400, f"Could not read {file.filename}: {type(e).__name__}: {e}")
    raws = obj if isinstance(obj, list) else [obj]
    raws = [r for r in raws if isinstance(r, dict) and "id" in r and "inner" in r]
    if not raws:
        raise HTTPException(400, "No ResPlan plans found (expected dicts with 'id', 'inner', room polygons …)")
    _add(raws, "uploaded")
    return {"added": len(raws), "ids": [int(r["id"]) for r in raws][:200]}


@app.get("/api/phases/{pid}")
def phases(pid: int):
    raw = _raw(pid)
    d = PHASE_DIR / str(pid)
    t0 = time.perf_counter()
    if not (d / "p9_final.png").exists():
        from walkthrough import render
        with _render_lock:                          # matplotlib is not thread-safe
            render(raw, d)
    return {"seconds": time.perf_counter() - t0,
            "images": [{"name": n, "title": t, "url": f"/phase-img/{pid}/{n}.png"}
                       for n, t in PHASES if (d / f"{n}.png").exists()]}


@app.get("/phase-img/{pid}/{name}.png")
def phase_img(pid: int, name: str):
    p = PHASE_DIR / str(pid) / f"{Path(name).name}.png"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
