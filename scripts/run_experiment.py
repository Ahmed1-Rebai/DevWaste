"""End-to-end synthetic experiment: simulate -> train -> evaluate -> compare.

Trains the LSTM Autoencoder on a "normal" run and evaluates it, together with
the baselines, on each anomaly scenario generated with *different* seeds (so
test noise never repeats training noise). Repeats for several training seeds
and reports mean and standard deviation over (training seed x test run).

Usage:
    python scripts/run_experiment.py [--out experiments] [--seeds 0 1 2]

Results are synthetic-data results only. They do not demonstrate
real-world accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from devwaste.data.simulator import SyntheticDataGenerator
from devwaste.models.score import evaluate_model
from devwaste.models.train import train_model

TEST_SCENARIOS = [
    "cpu-spike",
    "memory-growth",
    "network-spike",
    "oscillating",
    "subtle-oscillation",
]
DETECTORS = ["model", "zscore", "threshold", "rolling"]


def generate_data(
    data_dir: Path, n_train: int, n_test: int, test_runs: int
) -> tuple[Path, dict[str, list[Path]]]:
    """
    One normal training run plus ``test_runs`` independent runs per scenario.

    A held-out ``normal`` scenario measures the false-positive rate on clean
    data. Every run has its own seed, distinct from the training seed.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train_normal.csv"
    SyntheticDataGenerator(seed=1000).generate_scenario(
        "normal", n_samples=n_train
    ).to_csv(train_path, index=False)

    test_paths: dict[str, list[Path]] = {}
    for i, scenario in enumerate([*TEST_SCENARIOS, "normal"]):
        test_paths[scenario] = []
        for run in range(test_runs):
            path = data_dir / f"test_{scenario}_run{run}.csv"
            SyntheticDataGenerator(seed=2000 + 100 * i + run).generate_scenario(
                scenario, n_samples=n_test
            ).to_csv(path, index=False)
            test_paths[scenario].append(path)
    return train_path, test_paths


def detector_metrics(results: dict) -> dict[str, dict]:
    out = {"model": results["model_metrics"]}
    out.update(results["baseline_metrics"])
    return out


def summarise(runs: list[dict]) -> dict:
    """Mean/std of each metric per scenario and detector over seeds x test runs."""
    summary: dict = {}
    for scenario in [*TEST_SCENARIOS, "normal"]:
        summary[scenario] = {}
        for det in DETECTORS:
            per_seed = [m[det] for r in runs for m in r["scenarios"][scenario]]
            summary[scenario][det] = {
                key: {
                    "mean": float(np.mean([m[key] for m in per_seed])),
                    "std": float(np.std([m[key] for m in per_seed])),
                }
                for key in ("precision", "recall", "f1_score", "false_positive_rate")
            }
            aucs = [m["auc"] for m in per_seed if m["auc"] is not None]
            summary[scenario][det]["auc"] = (
                {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))}
                if aucs
                else None
            )
    return summary


def markdown_table(summary: dict) -> str:
    def cell(stat):
        return "n/a" if stat is None else f"{stat['mean']:.3f} ± {stat['std']:.3f}"

    lines = [
        "| Scenario | Detector | Precision | Recall | F1 | FPR | ROC-AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for scenario, dets in summary.items():
        for det, m in dets.items():
            name = "LSTM AE" if det == "model" else det
            lines.append(
                f"| {scenario} | {name} | {cell(m['precision'])} | {cell(m['recall'])} "
                f"| {cell(m['f1_score'])} | {cell(m['false_positive_rate'])} "
                f"| {cell(m['auc'])} |"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--test-runs", type=int, default=3, help="independent test runs per scenario"
    )
    args = parser.parse_args()

    train_path, test_paths = generate_data(
        args.out / "data", args.n_train, args.n_test, args.test_runs
    )
    runs = []
    for seed in args.seeds:
        model_dir = args.out / f"model_seed{seed}"
        start = time.perf_counter()
        train_results = train_model(
            train_path,
            model_dir,
            epochs=args.epochs,
            seed=seed,
            device="cpu",
            log=None,
        )
        train_seconds = time.perf_counter() - start
        scenarios, latency = {}, []
        for scenario, paths in test_paths.items():
            scenarios[scenario] = []
            for path in paths:
                results = evaluate_model(path, model_dir)
                metrics = detector_metrics(results)
                metrics["prevalence"] = results["anomaly_prevalence"]
                scenarios[scenario].append(metrics)
                latency.append(results["inference_ms_per_window"])
        runs.append(
            {
                "seed": seed,
                "train": train_results,
                "train_seconds": train_seconds,
                "inference_ms_per_window": float(np.mean(latency)),
                "scenarios": scenarios,
            }
        )
        print(
            f"seed {seed}: trained {train_results['epochs_trained']} epochs "
            f"in {train_seconds:.1f}s, threshold {train_results['threshold']:.4f}"
        )

    summary = summarise(runs)
    prevalence = {
        s: float(np.mean([m["prevalence"] for m in runs[0]["scenarios"][s]]))
        for s in summary
    }
    report = {
        "config": vars(args) | {"out": str(args.out)},
        "runs": runs,
        "prevalence": prevalence,
        "summary": summary,
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2, default=str))

    print()
    print(
        "Anomalous-window prevalence: "
        + ", ".join(f"{s} {p:.1%}" for s, p in prevalence.items())
    )
    print(
        f"Mean training time: {np.mean([r['train_seconds'] for r in runs]):.1f}s, "
        f"inference {np.mean([r['inference_ms_per_window'] for r in runs]):.4f} ms/window"
    )
    print()
    print(markdown_table(summary))
    print(f"\nFull results: {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
