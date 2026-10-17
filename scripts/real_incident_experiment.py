"""Controlled incidents in a real Docker container: collect -> train -> evaluate.

Starts one unprivileged, resource-limited container (``python:3.11-slim``,
1 CPU, 512 MB) that runs a light "normal" workload, then injects incidents
(CPU burn, memory leak, disk-write burst, CPU on/off oscillation). The
workload logs each incident's exact start/end, which become the labels.
Metrics come from Docker's streaming stats API, the same numbers
``devwaste collect`` uses.

The normal phase trains the model; the incident phase is the test set.
The container is created by this script and removed when it finishes; no
other container is touched.

Usage:
    python scripts/real_incident_experiment.py [--out experiments/real]

Requires a running Docker daemon and the python:3.11-slim image.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import docker
import numpy as np
import pandas as pd

from devwaste.collectors.docker_collector import DockerCollector
from devwaste.data.features import cumulative_to_deltas
from devwaste.models.score import evaluate_model
from devwaste.models.train import train_model

IMAGE = "python:3.11-slim"

# Runs inside the container. Prints one JSON line per incident boundary.
WORKLOAD = r"""
import json, os, random, sys, time
from datetime import datetime, timezone

random.seed(0)
NORMAL_SECONDS = int(sys.argv[1])
GAP = int(sys.argv[2])
buf = []

def now():
    return datetime.now(timezone.utc).isoformat()

def log(**kw):
    print(json.dumps(kw), flush=True)

def normal_tick():
    # light, slightly variable CPU and occasional small writes
    end = time.perf_counter() + random.uniform(0.01, 0.05)
    while time.perf_counter() < end:
        pass
    if random.random() < 0.1:
        with open("/tmp/normal.log", "ab") as f:
            f.write(os.urandom(32 * 1024))
    time.sleep(0.5)

def normal_for(seconds):
    end = time.time() + seconds
    while time.time() < end:
        normal_tick()

def cpu_burn(seconds):
    end = time.time() + seconds
    while time.time() < end:
        pass

def memory_leak(seconds):
    leak = []
    end = time.time() + seconds
    while time.time() < end:
        leak.append(bytearray(os.urandom(4 * 1024 * 1024)))  # ~4 MB/s
        normal_tick(); normal_tick()
    leak.clear()

def io_burst(seconds):
    end = time.time() + seconds
    with open("/tmp/burst.bin", "wb") as f:
        while time.time() < end:
            f.write(os.urandom(4 * 1024 * 1024))
            f.flush(); os.fsync(f.fileno())
            time.sleep(0.2)
    os.remove("/tmp/burst.bin")

def oscillate(seconds):
    end = time.time() + seconds
    while time.time() < end:
        cpu_burn(4)
        normal_for(4)

log(event="phase", phase="normal", t=now())
normal_for(NORMAL_SECONDS)
log(event="phase", phase="incidents", t=now())
for kind, fn, seconds in [
    ("cpu_burn", cpu_burn, 30),
    ("memory_leak", memory_leak, 60),
    ("io_burst", io_burst, 20),
    ("oscillation", oscillate, 64),
    ("cpu_burn", cpu_burn, 15),
]:
    normal_for(GAP)
    log(event="start", kind=kind, t=now())
    fn(seconds)
    log(event="end", kind=kind, t=now())
normal_for(GAP)
log(event="done", t=now())
"""


def stream_stats(
    container, collector: DockerCollector, rows: list, stop: threading.Event
):
    """Append one metric row per streamed stats sample (~1 s apart)."""
    for stats in container.stats(stream=True, decode=True):
        if stop.is_set():
            break
        if not stats.get("read") or stats["read"].startswith("0001"):
            continue
        rows.append(
            collector.metric_from_stats(container, stats, timestamp=stats["read"])
        )


def label_rows(df: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["label"] = 0
    df["incident"] = ""
    starts = {}
    for e in events:
        if e["event"] == "start":
            starts[e["kind"]] = pd.Timestamp(e["t"])
        elif e["event"] == "end":
            mask = (ts >= starts.pop(e["kind"])) & (ts <= pd.Timestamp(e["t"]))
            df.loc[mask, "label"] = 1
            df.loc[mask, "incident"] = e["kind"]
    return df


def run_container(normal_seconds: int, gap: int) -> tuple[pd.DataFrame, list[dict]]:
    client = docker.from_env()
    collector = DockerCollector(client=client)
    container = client.containers.run(
        IMAGE,
        ["python", "-u", "-c", WORKLOAD, str(normal_seconds), str(gap)],
        detach=True,
        name=f"devwaste-incident-exp-{int(time.time())}",
        labels={"devwaste.experiment": "incidents"},
        nano_cpus=1_000_000_000,
        mem_limit="512m",
        network_mode="none",
        read_only=False,
        privileged=False,
    )
    rows: list = []
    stop = threading.Event()
    thread = threading.Thread(
        target=stream_stats, args=(container, collector, rows, stop), daemon=True
    )
    try:
        thread.start()
        result = container.wait()
        stop.set()
        thread.join(timeout=5)
        if result["StatusCode"] != 0:
            raise RuntimeError(container.logs().decode())
        events = [
            json.loads(line)
            for line in container.logs(stdout=True, stderr=False).splitlines()
        ]
    finally:
        container.remove(force=True)  # only the container created above

    df = pd.DataFrame([vars(r) for r in rows])
    df["state"] = "running"
    return df, events


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("experiments/real"))
    parser.add_argument("--normal-seconds", type=int, default=300)
    parser.add_argument("--gap", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Running workload (~{args.normal_seconds + 6 * args.gap + 189}s)...")
    raw, events = run_container(args.normal_seconds, args.gap)
    raw.to_csv(args.out / "raw_stats.csv", index=False)
    (args.out / "events.json").write_text(json.dumps(events, indent=2))

    df = label_rows(cumulative_to_deltas(raw), events)
    split = pd.Timestamp(next(e["t"] for e in events if e.get("phase") == "incidents"))
    ts = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    train_df, test_df = df[ts < split], df[ts >= split]
    train_df.to_csv(args.out / "train.csv", index=False)
    test_df.to_csv(args.out / "test.csv", index=False)
    print(
        f"Samples: train {len(train_df)}, test {len(test_df)} "
        f"({test_df['label'].mean():.1%} labelled incident samples)"
    )

    detectors: dict[str, list[dict]] = {}
    for seed in args.seeds:
        model_dir = args.out / f"model_seed{seed}"
        train_model(
            args.out / "train.csv", model_dir, seed=seed, device="cpu", log=None
        )
        results = evaluate_model(args.out / "test.csv", model_dir, top_k=0)
        detectors.setdefault("LSTM AE", []).append(results["model_metrics"])
        for name, m in results["baseline_metrics"].items():
            detectors.setdefault(name, []).append(m)
        prevalence = results["anomaly_prevalence"]

    print(f"\nAnomalous-window prevalence: {prevalence:.1%}\n")
    print("| Detector | Precision | Recall | F1 | FPR | ROC-AUC |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for name, runs in detectors.items():
        stats = {}
        for key in ("precision", "recall", "f1_score", "false_positive_rate", "auc"):
            values = [r[key] for r in runs if r[key] is not None]
            stats[key] = (
                (float(np.mean(values)), float(np.std(values))) if values else None
            )
        summary[name] = stats
        cells = [
            "n/a" if stats[k] is None else f"{stats[k][0]:.3f} ± {stats[k][1]:.3f}"
            for k in ("precision", "recall", "f1_score", "false_positive_rate", "auc")
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")

    (args.out / "results.json").write_text(
        json.dumps({"events": events, "summary": summary}, indent=2)
    )
    print(f"\nFull results: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
