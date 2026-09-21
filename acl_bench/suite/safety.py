"""Keep a long multi-process run from hurting the machine (macOS).

A CPU-bound job cannot physically damage a Mac, since it throttles itself. The real
risks are a hot, loud, unresponsive machine, a drained battery, a full disk and
runaway memory, so `run_jobs` protects against those:

  - workers run at reduced CPU priority (`nice`), so the interactive desktop wins;
  - a watchdog is consulted before every new job is dispatched; if the machine is
    thermally throttled, low on free memory, on battery, or short on disk, dispatch
    PAUSES (jobs already running finish) and resumes when it clears;
  - a stop file (`touch <file>`) stops dispatching new jobs and exits cleanly;
  - one failed job is logged and skipped instead of aborting a multi-hour run.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def on_ac_power() -> bool | None:
    """True on AC, False on battery, None if unknown (no `pmset`, or a desktop)."""
    out = _run(["pmset", "-g", "batt"])
    if "'AC Power'" in out:
        return True
    if "'Battery Power'" in out:
        return False
    return None


def thermal_speed_limit() -> int | None:
    """macOS's CPU speed limit in percent (100 = unthrottled), or None if no throttling
    has been recorded."""
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", _run(["pmset", "-g", "therm"]))
    return int(m.group(1)) if m else None


def free_memory_percent() -> int | None:
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", _run(["memory_pressure"]))
    return int(m.group(1)) if m else None


def free_disk_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


@dataclass
class Limits:
    min_free_memory_pct: int = 15
    min_cpu_speed_limit: int = 90        # below this macOS is thermally throttling
    require_ac: bool = True
    min_free_disk_gb: float = 5.0


class Watchdog:
    """`status()` returns (ok, reason). Probes are injectable so the logic is testable."""

    def __init__(self, limits: Limits | None = None, disk_path: str = ".", check_every: float = 15.0,
                 probes: dict[str, Callable] | None = None, clock: Callable[[], float] = time.monotonic):
        self.limits, self.disk_path, self.check_every, self.clock = limits or Limits(), disk_path, check_every, clock
        self.probes = {"ac": on_ac_power, "speed": thermal_speed_limit, "memory": free_memory_percent,
                       "disk": lambda: free_disk_gb(disk_path), **(probes or {})}
        self._cached: tuple[bool, str] = (True, "ok")
        self._at = -1e18

    def status(self) -> tuple[bool, str]:
        if self.clock() - self._at < self.check_every:
            return self._cached
        lim, p = self.limits, self.probes
        ok, reason = True, "ok"
        speed, mem, ac, disk = p["speed"](), p["memory"](), p["ac"](), p["disk"]()
        if speed is not None and speed < lim.min_cpu_speed_limit:
            ok, reason = False, f"thermal throttling (CPU speed limit {speed}%)"
        elif mem is not None and mem < lim.min_free_memory_pct:
            ok, reason = False, f"low free memory ({mem}%)"
        elif lim.require_ac and ac is False:
            ok, reason = False, "running on battery"
        elif disk is not None and disk < lim.min_free_disk_gb:
            ok, reason = False, f"low free disk ({disk:.1f} GB)"
        self._cached, self._at = (ok, reason), self.clock()
        return self._cached


def _init_worker(niceness: int) -> None:
    if niceness:
        os.nice(niceness)


def run_jobs(jobs: list, job_fn: Callable, workers: int, on_result: Callable, watchdog: Watchdog,
             niceness: int = 10, stop_file: str | None = None, log: Callable = print,
             poll: float = 0.5) -> list:
    """Run `job_fn(job)` over a process pool with the safeguards above. `on_result` is
    called with each job's return value in the parent. Returns the jobs that failed."""
    for var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")            # one thread per worker; inherited by spawned workers
    queue, pending, failed, paused_reason = list(jobs), [], [], None
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(niceness,)) as pool:
        while queue or pending:
            still = []
            for job, result in pending:
                if not result.ready():
                    still.append((job, result))
                    continue
                try:
                    on_result(result.get())
                except Exception as exc:                # one bad job must not kill a multi-hour run
                    failed.append(job)
                    log(f"job failed and was skipped: {job!r}: {type(exc).__name__}: {exc}")
            pending = still
            if queue and stop_file and os.path.exists(stop_file):
                log(f"stop file {stop_file} found: not starting {len(queue)} remaining jobs")
                queue.clear()
            ok, reason = watchdog.status()
            if ok:
                if paused_reason:
                    log(f"resumed (was paused: {paused_reason})")
                    paused_reason = None
                while queue and len(pending) < workers:
                    job = queue.pop(0)
                    pending.append((job, pool.apply_async(job_fn, (job,))))
            elif queue and reason != paused_reason:
                paused_reason = reason
                log(f"PAUSED, no new jobs until this clears: {reason}")
            time.sleep(poll)
    return failed
