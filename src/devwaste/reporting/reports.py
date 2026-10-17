"""Terminal and JSON report rendering."""

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_json_report(data: dict[str, Any], path: Path):
    """Write a JSON report, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def print_container_table(console: Console, containers: list[dict[str, Any]]):
    """
    Print container information as a table.

    Args:
        console: Rich console
        containers: Rows as produced by ``devwaste scan``
    """
    if not containers:
        console.print("[yellow]No containers found[/yellow]")
        return

    table = Table(title="Docker Containers")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("State", style="yellow")
    table.add_column("Image", style="blue")
    table.add_column("CPU %", justify="right")
    table.add_column("Mem %", justify="right")
    table.add_column("Mem used", justify="right")

    for c in containers:
        metrics = c.get("metrics") or {}
        mem_mb = metrics.get("memory_usage")
        table.add_row(
            c["id"][:12],
            c["name"],
            c["state"],
            c["image"],
            _fmt(metrics.get("cpu_percent"), 1),
            _fmt(metrics.get("memory_percent"), 1),
            f"{mem_mb / 1024**2:.0f} MB" if mem_mb is not None else "n/a",
        )

    console.print(table)


def print_training_summary(console: Console, results: dict[str, Any], output_dir: Path):
    """Print the result of ``train_model``."""
    console.print("[green]Training complete[/green]")
    console.print(f"  Epochs trained:      {results['epochs_trained']}")
    console.print(f"  Final train loss:    {results['final_train_loss']:.4f}")
    console.print(f"  Best val loss:       {results['best_val_loss']:.4f}")
    console.print(
        f"  Threshold:           {results['threshold']:.4f} ({results['threshold_method']})"
    )
    console.print(
        f"  Windows (train/val): {results['n_train_windows']}/{results['n_val_windows']}"
    )
    console.print(f"  Artifacts saved to:  {output_dir}")


def print_evaluation(console: Console, results: dict[str, Any]):
    """Print the result of ``evaluate_model``."""
    console.print(
        f"Scored {results['n_windows']} windows: {results['n_flagged']} flagged "
        f"({results['flagged_fraction']:.1%}) at threshold {results['threshold']:.4f}"
    )
    console.print(f"Inference: {results['inference_ms_per_window']:.3f} ms/window")

    if results["labelled"]:
        console.print(
            f"Anomalous window prevalence: {results['anomaly_prevalence']:.1%}"
        )
        table = Table(title="Window-level detection metrics")
        for col in ("Detector", "Precision", "Recall", "F1", "FPR", "ROC-AUC"):
            table.add_column(col, justify="left" if col == "Detector" else "right")

        rows = [("LSTM Autoencoder", results["model_metrics"])]
        rows += [(f"baseline: {k}", v) for k, v in results["baseline_metrics"].items()]
        for name, m in rows:
            table.add_row(
                name,
                _fmt(m["precision"]),
                _fmt(m["recall"]),
                _fmt(m["f1_score"]),
                _fmt(m["false_positive_rate"]),
                _fmt(m["auc"]),
            )
        console.print(table)
    else:
        console.print(
            "[yellow]No 'label' column: detection only, no accuracy metrics.[/yellow]"
        )

    if results["top_windows"]:
        table = Table(title="Highest-scoring windows")
        meta_cols = [
            c
            for c in ("container_name", "run_id", "start_timestamp", "end_timestamp")
            if c in results["top_windows"][0]
        ]
        for col in [*meta_cols, "score", "flagged"]:
            table.add_column(col)
        for row in results["top_windows"]:
            table.add_row(
                *[_fmt(row.get(c), 4) for c in [*meta_cols, "score", "flagged"]]
            )
        console.print(table)

    console.print(
        "[dim]Diagnostic signal only: a high score means the pattern differs from "
        "training data, not that a container is wasting resources.[/dim]"
    )
