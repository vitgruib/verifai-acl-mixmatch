"""Failure records: one directory per falsification run, written so every counterexample
can be replayed and diagnosed later without the machine that found it.

  run.json       header: what was falsified (env, algorithm, checkpoint, training config),
                 how (space, sampler and its params, spec, budget, seeds), code versions
  samples.jsonl  one line per proposed point, valid or not, in proposal order (fields below)
  traces.npz     agent trajectories for counterexamples only (key "i<index>")
  summary.json   aggregates (written by `Recorder.close`)

Sample fields: i, t (seconds since start), params (the falsifier's raw point), valid,
invalid_reason, level (replayable encoding), desc (interpretable features), returns and
lengths (per attempt), solve_rate, mean_return, rho (spec robustness; < 0 = violated), cex,
hard (solve_rate == 0), train (where the level sits relative to training: see the env's
`train_context`). The full schema and how to read it: docs/failure_records.md.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import numpy as np

SCHEMA_VERSION = 2   # 2: per-row "oracle" (solvability certificate), header "oracle"


def git_commit(path: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Recorder:
    def __init__(self, out_dir: str, header: dict, desc_keys: tuple):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.desc_keys = out_dir, desc_keys
        self.header = {"schema_version": SCHEMA_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "host": platform.node(), **header}
        with open(os.path.join(out_dir, "run.json"), "w") as f:
            json.dump(self.header, f, indent=1, default=_json)
        self._f = open(os.path.join(out_dir, "samples.jsonl"), "w")
        self._traces, self._rows, self.t0 = {}, [], time.time()

    def add(self, row: dict, trace: dict | None = None):
        row = {"i": len(self._rows), "t": round(time.time() - self.t0, 3), **row}
        self._f.write(json.dumps(row, default=_json) + "\n")
        self._rows.append(row)
        if trace is not None and row.get("cex"):
            for k, v in trace.items():
                self._traces[f"i{row['i']}_{k}"] = v

    def close(self) -> dict:
        self._f.close()
        if self._traces:
            np.savez_compressed(os.path.join(self.out_dir, "traces.npz"), **self._traces)
        s = summarize(self._rows, self.desc_keys)
        s["wall_s"] = round(time.time() - self.t0, 1)
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(s, f, indent=1, default=_json)
        return s


def summarize(rows: list, desc_keys: tuple) -> dict:
    valid = [r for r in rows if r["valid"]]
    cex = [r for r in valid if r["cex"]]
    ok = [r for r in valid if not r["cex"]]

    def mean(rs, f):
        v = [f(r) for r in rs if f(r) is not None]
        return round(float(np.mean(v)), 4) if v else None

    first = cex[0] if cex else None
    # distinct failure modes: counterexamples binned coarsely on their features
    cells = {tuple(None if r["desc"][k] is None else int(r["desc"][k]) // 4 for k in desc_keys) for r in cex}
    train_keys = sorted({k for r in valid for k in (r.get("train") or {})})
    return {
        "n": len(rows), "n_valid": len(valid), "invalid_rate": round(1 - len(valid) / max(len(rows), 1), 4),
        "n_cex": len(cex), "cex_rate": round(len(cex) / max(len(valid), 1), 4),
        "n_hard": sum(r["hard"] for r in valid),
        "first_cex_i": first and first["i"], "first_cex_t": first and first["t"],
        "min_rho": min((r["rho"] for r in valid), default=None),
        "mean_rho": mean(valid, lambda r: r["rho"]),
        "distinct_cex_cells": len(cells),
        "desc_cex": {k: mean(cex, lambda r, k=k: r["desc"][k]) for k in desc_keys},
        "desc_ok": {k: mean(ok, lambda r, k=k: r["desc"][k]) for k in desc_keys},
        "train_cex": {k: mean(cex, lambda r, k=k: (r.get("train") or {}).get(k)) for k in train_keys},
        "train_ok": {k: mean(ok, lambda r, k=k: (r.get("train") or {}).get(k)) for k in train_keys},
    }


def load(run_dir: str):
    """(header, rows, summary or None) for a run directory."""
    with open(os.path.join(run_dir, "run.json")) as f:
        header = json.load(f)
    with open(os.path.join(run_dir, "samples.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    p = os.path.join(run_dir, "summary.json")
    summary = json.load(open(p)) if os.path.exists(p) else None
    return header, rows, summary


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
