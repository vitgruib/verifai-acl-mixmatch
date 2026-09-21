import os
import sys
import time

import pandas as pd
import pytest

from acl_bench.suite import safety
from acl_bench.suite.run_arms import append_rows
from acl_bench.suite.safety import Limits, Watchdog, run_jobs


# ---- job functions (module level so spawned workers can import them) ----
def _timed_job(x):
    start = time.time()
    time.sleep(0.4)
    return (x, start, time.time())


def _nice_job(x):
    return os.nice(0)


def _failing_job(x):
    if x == 2:
        raise RuntimeError("boom")
    return x


# ---- probes parse macOS output ----
def test_probes_parse_pmset_and_memory_output(monkeypatch):
    outputs = {}
    monkeypatch.setattr(safety, "_run", lambda cmd: outputs.get(" ".join(cmd), ""))
    outputs["pmset -g batt"] = "Now drawing from 'AC Power'\n -InternalBattery-0 80%; charging"
    assert safety.on_ac_power() is True
    outputs["pmset -g batt"] = "Now drawing from 'Battery Power'\n"
    assert safety.on_ac_power() is False
    outputs["pmset -g batt"] = ""
    assert safety.on_ac_power() is None
    outputs["pmset -g therm"] = "CPU_Scheduler_Limit = 100\nCPU_Available_CPUs = 10\nCPU_Speed_Limit = 70\n"
    assert safety.thermal_speed_limit() == 70
    outputs["pmset -g therm"] = "Note: No thermal warning level has been recorded\n"
    assert safety.thermal_speed_limit() is None
    outputs["memory_pressure"] = "The system has 25769803776 (1572864 pages)\nSystem-wide memory free percentage: 42%\n"
    assert safety.free_memory_percent() == 42


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS probes")
def test_real_probes_return_sane_values_on_this_machine():
    assert safety.on_ac_power() in (True, False, None)
    assert safety.thermal_speed_limit() is None or 0 < safety.thermal_speed_limit() <= 100
    assert safety.free_memory_percent() is None or 0 <= safety.free_memory_percent() <= 100
    assert safety.free_disk_gb(".") > 0


# ---- watchdog decisions ----
def dog(**probes):
    base = {"ac": lambda: True, "speed": lambda: None, "memory": lambda: 60, "disk": lambda: 100.0}
    return Watchdog(Limits(), check_every=0, probes={**base, **probes})


def test_watchdog_flags_each_hazard():
    assert dog().status() == (True, "ok")
    assert "thermal" in dog(speed=lambda: 60).status()[1]
    assert "memory" in dog(memory=lambda: 5).status()[1]
    assert "battery" in dog(ac=lambda: False).status()[1]
    assert "disk" in dog(disk=lambda: 1.0).status()[1]


def test_battery_is_allowed_only_when_asked():
    wd = Watchdog(Limits(require_ac=False), check_every=0, probes={"ac": lambda: False, "speed": lambda: None,
                                                                    "memory": lambda: 60, "disk": lambda: 100.0})
    assert wd.status()[0] is True


def test_watchdog_does_not_probe_more_often_than_asked():
    calls = []
    clock = [0.0]
    wd = Watchdog(Limits(), check_every=15, clock=lambda: clock[0],
                  probes={"ac": lambda: calls.append(1) or True, "speed": lambda: None,
                          "memory": lambda: 60, "disk": lambda: 100.0})
    wd.status(); wd.status(); clock[0] = 10; wd.status()
    assert len(calls) == 1
    clock[0] = 20; wd.status()
    assert len(calls) == 2


# ---- the guarded pool ----
def test_pool_runs_every_job_never_exceeds_workers_and_lowers_priority():
    results = []
    run_jobs(list(range(6)), _timed_job, workers=2, on_result=results.append, watchdog=dog(),
             niceness=5, poll=0.05, log=lambda m: None)
    assert sorted(r[0] for r in results) == list(range(6))
    events = sorted([(s, 1) for _, s, _ in results] + [(e, -1) for _, _, e in results])
    live = peak = 0
    for _, d in events:
        live += d
        peak = max(peak, live)
    assert peak <= 2
    nice = []
    run_jobs([0, 1], _nice_job, workers=2, on_result=nice.append, watchdog=dog(), niceness=7, poll=0.05,
             log=lambda m: None)
    assert all(n >= os.nice(0) + 7 for n in nice)


def test_dispatch_pauses_while_unhealthy_then_resumes():
    seen, log = {"n": 0}, []

    def flaky():
        seen["n"] += 1
        return 40 if seen["n"] <= 6 else None            # thermally throttled for the first 6 checks

    started = time.time()
    results = []
    run_jobs(list(range(3)), _timed_job, workers=3, on_result=results.append, watchdog=dog(speed=flaky),
             niceness=0, poll=0.1, log=log.append)
    assert len(results) == 3
    assert any("PAUSED" in m and "thermal" in m for m in log) and any("resumed" in m for m in log)
    assert min(s for _, s, _ in results) - started >= 0.5    # nothing started during the pause


def test_stop_file_prevents_new_jobs(tmp_path):
    stop = tmp_path / "STOP"
    stop.write_text("")
    results, log = [], []
    run_jobs(list(range(5)), _timed_job, workers=2, on_result=results.append, watchdog=dog(),
             stop_file=str(stop), niceness=0, poll=0.05, log=log.append)
    assert results == [] and any("stop file" in m for m in log)


def test_one_failed_job_is_skipped_not_fatal():
    results, log = [], []
    failed = run_jobs([0, 1, 2, 3], _failing_job, workers=2, on_result=results.append, watchdog=dog(),
                      niceness=0, poll=0.05, log=log.append)
    assert failed == [2] and sorted(results) == [0, 1, 3] and any("boom" in m for m in log)


# ---- results are appended, not rewritten ----
def test_append_rows_writes_header_once_and_rejects_mismatched_columns(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows(path, [{"arm": "N", "seed": 1, "v": 1.0}])
    append_rows(path, [{"arm": "A", "seed": 1, "v": 2.0}])
    assert open(path).read().count("arm,seed,v") == 1
    assert len(pd.read_csv(path)) == 2
    with pytest.raises(ValueError):
        append_rows(path, [{"arm": "B", "other": 3}])
