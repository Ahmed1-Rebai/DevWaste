"""Command-line interface for DevWaste AI."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from devwaste import __version__
from devwaste.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_DURATION,
    DEFAULT_COLLECTION_INTERVAL,
    DEFAULT_DB_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_THRESHOLD_QUANTILE,
    DEFAULT_WINDOW_SIZE,
)

app = typer.Typer(
    name="devwaste",
    help="Deep Learning for Docker Resource Anomaly Detection (diagnostic only).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"DevWaste AI version {__version__}")
        raise typer.Exit()


@app.callback()
def common(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
):
    """Deep Learning for Docker Resource Anomaly Detection (diagnostic only)."""


def _docker_client_or_exit():
    """Connect to Docker or exit with a helpful message."""
    from devwaste.collectors.docker_collector import DockerCollector

    try:
        return DockerCollector()
    except ImportError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e
    except RuntimeError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        err_console.print(
            "Make sure the Docker daemon is running and you have access to it."
        )
        raise typer.Exit(1) from e


def _container_row(collector, container, with_metrics: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": container.id,
        "name": container.name,
        "state": container.status,
        "image": container.attrs.get("Config", {}).get("Image", "unknown"),
        "metrics": None,
    }
    if with_metrics and container.status == "running":
        metric = collector.collect_metrics(container)
        if metric is not None:
            row["metrics"] = {
                "cpu_percent": metric.cpu_percent,
                "memory_usage": metric.memory_usage,
                "memory_limit": metric.memory_limit,
                "memory_percent": metric.memory_percent,
                "restart_count": metric.restart_count,
            }
    return row


@app.command()
def scan(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    all_containers: bool = typer.Option(
        False, "--all", "-a", help="Include stopped containers"
    ),
    no_stats: bool = typer.Option(
        False, "--no-stats", help="Skip the CPU/memory snapshot (faster)"
    ),
):
    """List Docker containers with a CPU/memory snapshot (read-only)."""
    from devwaste.reporting.reports import print_container_table

    collector = _docker_client_or_exit()
    containers = collector.list_containers(running_only=not all_containers)
    rows = [_container_row(collector, c, with_metrics=not no_stats) for c in containers]

    if json_output:
        typer.echo(json.dumps({"status": "success", "containers": rows}, indent=2))
    else:
        print_container_table(console, rows)


@app.command()
def collect(
    interval: int = typer.Option(
        DEFAULT_COLLECTION_INTERVAL,
        "--interval",
        min=1,
        help="Seconds between collection rounds",
    ),
    duration: int = typer.Option(
        DEFAULT_COLLECTION_DURATION,
        "--duration",
        min=1,
        help="Total collection time in seconds",
    ),
    db: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="SQLite database path"),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Also export this session to CSV (byte counters converted to per-interval deltas)",
    ),
):
    """Collect metrics from running Docker containers into SQLite."""
    from rich.progress import BarColumn, Progress, TextColumn

    from devwaste.data.features import cumulative_to_deltas
    from devwaste.storage.sqlite_store import SQLiteStore

    collector = _docker_client_or_exit()
    rounds = max(1, duration // interval)
    session_start = datetime.now(UTC).isoformat()
    n_saved = 0

    console.print(
        f"Collecting every {interval}s for {duration}s ({rounds} rounds) into {db}"
    )
    with SQLiteStore(db) as storage:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting", total=rounds)
            for i in range(rounds):
                round_start = time.monotonic()
                metrics = collector.collect_all_metrics(running_only=True)
                storage.save_metrics(metrics)
                n_saved += len(metrics)
                progress.update(task, advance=1)

                if i < rounds - 1:
                    # stats calls take ~1-2s each; keep rounds `interval` apart
                    time.sleep(max(0.0, interval - (time.monotonic() - round_start)))

        console.print(
            f"[green]Saved {n_saved} observations from {rounds} rounds[/green]"
        )

        if output:
            df = storage.load_metrics(start_time=session_start)
            df = cumulative_to_deltas(df.drop(columns=["id"]))
            output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output, index=False)
            console.print(f"[green]Exported {len(df)} rows to {output}[/green]")


@app.command()
def simulate(
    scenario: str = typer.Option(
        ...,
        "--scenario",
        help="normal, cpu-spike, memory-growth, network-spike, or oscillating",
    ),
    output: Path = typer.Option(..., "--output", help="Output CSV path"),
    n_samples: int = typer.Option(1000, "--n-samples", min=1, help="Number of samples"),
    interval: float = typer.Option(
        1.0, "--interval", min=0.001, help="Seconds between samples"
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Random seed"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Run identifier (default: <scenario>-seed<seed>)"
    ),
):
    """Generate a labelled synthetic time-series dataset (no Docker needed)."""
    from devwaste.data.simulator import SyntheticDataGenerator

    try:
        df = SyntheticDataGenerator(seed=seed).generate_scenario(
            scenario=scenario, n_samples=n_samples, interval=interval, run_id=run_id
        )
    except ValueError as e:
        err_console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.print(
        f"[green]Generated '{scenario}' (seed {seed}): {n_samples} samples, "
        f"{int(df['label'].sum())} labelled anomalous -> {output}[/green]"
    )


@app.command()
def train(
    data: Path = typer.Option(..., "--data", help="Training data CSV"),
    output_dir: Path = typer.Option(
        DEFAULT_ARTIFACTS_DIR, "--output-dir", help="Artifacts directory"
    ),
    window_size: int = typer.Option(
        DEFAULT_WINDOW_SIZE, "--window-size", min=2, help="Sequence length"
    ),
    hidden_dim: int = typer.Option(
        DEFAULT_HIDDEN_DIM, "--hidden-dim", min=1, help="LSTM hidden dimension"
    ),
    num_layers: int = typer.Option(1, "--num-layers", min=1, help="LSTM layers"),
    epochs: int = typer.Option(
        DEFAULT_EPOCHS, "--epochs", min=1, help="Maximum training epochs"
    ),
    batch_size: int = typer.Option(
        DEFAULT_BATCH_SIZE, "--batch-size", min=1, help="Batch size"
    ),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE, "--learning-rate", help="Learning rate"
    ),
    threshold_quantile: float = typer.Option(
        DEFAULT_THRESHOLD_QUANTILE,
        "--threshold-quantile",
        min=0.5,
        max=1.0,
        help="Quantile of normal validation errors used as threshold",
    ),
    threshold_method: str = typer.Option(
        "quantile",
        "--threshold-method",
        help="Calibration method: 'quantile' or 'evt' (Extreme Value Theory / POT)",
    ),
    validation_split: float = typer.Option(
        0.2,
        "--validation-split",
        min=0.05,
        max=0.5,
        help="Fraction of each series for validation",
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Random seed"),
    device: str | None = typer.Option(
        None, "--device", help="cpu or cuda (default: auto)"
    ),
):
    """Train the LSTM Autoencoder on normal behaviour."""
    from devwaste.models.train import train_model
    from devwaste.reporting.reports import print_training_summary

    if not data.exists():
        err_console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)

    try:
        results = train_model(
            data_path=data,
            output_dir=output_dir,
            window_size=window_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            threshold_quantile=threshold_quantile,
            threshold_method=threshold_method,
            validation_split=validation_split,
            seed=seed,
            device=device,
            log=console.print,
        )
    except ValueError as e:
        err_console.print(f"[red]Error during training: {e}[/red]")
        raise typer.Exit(1) from e

    print_training_summary(console, results, output_dir)


@app.command()
def evaluate(
    data: Path = typer.Option(
        ..., "--data", help="Data CSV to score (labelled or not)"
    ),
    model: Path = typer.Option(
        DEFAULT_ARTIFACTS_DIR, "--model", help="Model artifacts directory"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="Write the full JSON report here"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the JSON report instead of tables"
    ),
    top_k: int = typer.Option(
        10, "--top", min=0, help="Number of highest-scoring windows to show"
    ),
    consecutive: int = typer.Option(
        1,
        "--consecutive",
        min=1,
        help="Minimum consecutive anomalous windows required for an alert",
    ),
    recovery_margin: float = typer.Option(
        0.0,
        "--recovery-margin",
        min=0.0,
        help="Seconds after an incident to treat as recovery transition",
    ),
):
    """Score data with a trained model; compare against baselines if labelled."""
    from devwaste.models.score import evaluate_model
    from devwaste.reporting.reports import print_evaluation, write_json_report

    if not data.exists():
        err_console.print(f"[red]Error: Data file not found: {data}[/red]")
        raise typer.Exit(1)
    if not model.is_dir():
        err_console.print(f"[red]Error: Model directory not found: {model}[/red]")
        raise typer.Exit(1)

    try:
        results = evaluate_model(
            data_path=data,
            model_dir=model,
            top_k=top_k,
            consecutive=consecutive,
            recovery_margin=recovery_margin,
        )
    except (ValueError, FileNotFoundError) as e:
        err_console.print(f"[red]Error during evaluation: {e}[/red]")
        raise typer.Exit(1) from e

    if output:
        write_json_report(results, output)
    if json_output:
        typer.echo(json.dumps(results, indent=2, default=str))
    else:
        print_evaluation(console, results)
        if output:
            console.print(f"Report written to {output}")


@app.command()
def report(
    input_file: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to evaluation JSON report produced by 'devwaste evaluate'",
    ),
    data: Path | None = typer.Option(
        None, "--data", help="Data CSV to score and report"
    ),
    model: Path = typer.Option(
        DEFAULT_ARTIFACTS_DIR, "--model", help="Model artifacts directory"
    ),
    output: Path = typer.Option(
        Path("reports/report.html"), "--output", "-o", help="Output HTML file path"
    ),
    consecutive: int = typer.Option(
        1, "--consecutive", min=1, help="Consecutive anomalous windows required"
    ),
    recovery_margin: float = typer.Option(
        0.0, "--recovery-margin", min=0.0, help="Recovery margin in seconds"
    ),
):
    """Generate a self-contained offline HTML visual diagnostic report."""
    from devwaste.models.score import evaluate_model
    from devwaste.reporting.html_report import write_html_report

    if input_file is not None:
        if not input_file.exists():
            err_console.print(f"[red]Error: Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        try:
            results = json.loads(input_file.read_text(encoding="utf-8"))
        except Exception as e:
            err_console.print(f"[red]Error parsing JSON input: {e}[/red]")
            raise typer.Exit(1) from e
    elif data is not None:
        if not data.exists():
            err_console.print(f"[red]Error: Data file not found: {data}[/red]")
            raise typer.Exit(1)
        if not model.is_dir():
            err_console.print(f"[red]Error: Model directory not found: {model}[/red]")
            raise typer.Exit(1)
        try:
            results = evaluate_model(
                data_path=data,
                model_dir=model,
                consecutive=consecutive,
                recovery_margin=recovery_margin,
            )
        except Exception as e:
            err_console.print(f"[red]Error during evaluation: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        err_console.print(
            "[red]Error: Must provide either --input <report.json> or --data <data.csv>[/red]"
        )
        raise typer.Exit(1)

    try:
        write_html_report(results, output)
        console.print(f"[green]Visual diagnostic report written to {output}[/green]")
    except Exception as e:
        err_console.print(f"[red]Error generating HTML report: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def monitor(
    model: Path = typer.Option(
        DEFAULT_ARTIFACTS_DIR, "--model", help="Model artifacts directory"
    ),
    interval: int = typer.Option(
        DEFAULT_COLLECTION_INTERVAL,
        "--interval",
        min=1,
        help="Seconds between collection and scoring rounds",
    ),
    duration: int | None = typer.Option(
        None,
        "--duration",
        min=1,
        help="Total monitoring time in seconds (default: run until stopped)",
    ),
    db: Path | None = typer.Option(
        None, "--db", help="Optional SQLite database to persist metrics"
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional file path to append anomaly alerts/events as JSON lines",
    ),
    consecutive: int = typer.Option(
        1,
        "--consecutive",
        min=1,
        help="Consecutive anomalous windows required for an alert",
    ),
    threshold_multiplier: float = typer.Option(
        1.0,
        "--threshold-multiplier",
        min=0.1,
        help="Multiplier on the model threshold",
    ),
    adaptive_rate: float = typer.Option(
        0.0,
        "--adaptive-rate",
        min=0.0,
        max=0.5,
        help="Exponential adaptation rate for threshold drift on normal windows (0.0 to disable)",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Stream alerts and rounds in JSON format"
    ),
):
    """Monitor running Docker containers in real-time with a trained model."""
    from devwaste.models.monitor import OnlineMonitor
    from devwaste.reporting.reports import print_monitor_round
    from devwaste.storage.sqlite_store import SQLiteStore

    if not model.is_dir():
        err_console.print(f"[red]Error: Model directory not found: {model}[/red]")
        raise typer.Exit(1)

    collector = _docker_client_or_exit()

    try:
        online_mon = OnlineMonitor(
            model_dir=model,
            threshold_multiplier=threshold_multiplier,
            consecutive=consecutive,
            adaptive_rate=adaptive_rate,
        )
    except (ValueError, FileNotFoundError) as e:
        err_console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1) from e

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)

    storage = SQLiteStore(db) if db else None
    start_time = time.monotonic()

    if not json_output:
        console.print(
            f"Starting real-time monitoring with model {model} (interval: {interval}s, "
            f"window: {online_mon.window_size}, threshold: {online_mon.threshold:.4f})"
        )
        console.print("Press Ctrl+C to stop.")

    try:
        while True:
            if duration is not None and (time.monotonic() - start_time) >= duration:
                break
            round_start = time.monotonic()
            metrics = collector.collect_all_metrics(running_only=True)
            if storage and metrics:
                storage.save_metrics(metrics)

            results = online_mon.process_round(metrics)

            if json_output:
                for r in results:
                    if r.get("status") == "ready":
                        typer.echo(json.dumps(r, default=str))
            else:
                print_monitor_round(console, results)

            if output:
                with output.open("a", encoding="utf-8") as f:
                    for r in results:
                        if r.get("is_anomalous"):
                            f.write(json.dumps(r, default=str) + "\n")

            elapsed = time.monotonic() - round_start
            sleep_sec = max(0.0, interval - elapsed)
            if (
                duration is not None
                and (time.monotonic() - start_time) + sleep_sec >= duration
            ):
                time.sleep(max(0.0, duration - (time.monotonic() - start_time)))
                break
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        if not json_output:
            console.print("\n[yellow]Monitoring stopped by user.[/yellow]")
    finally:
        if storage:
            storage.close()


@app.command()
def cleanup(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Only show recommendations (the only supported mode)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Show read-only cleanup recommendations for Docker resources."""
    from devwaste.safety.cleanup import CleanupAdvisor

    if not dry_run:
        err_console.print(
            "[red]Refusing to run: DevWaste never performs cleanup itself.[/red]\n"
            "Run with --dry-run, review the recommendations, and execute the "
            "suggested commands manually if you agree with them."
        )
        raise typer.Exit(2)

    advisor = CleanupAdvisor(_docker_client_or_exit().client)
    recommendations = advisor.get_recommendations()

    if json_output:
        typer.echo(
            json.dumps({"dry_run": True, "recommendations": recommendations}, indent=2)
        )
    else:
        console.print(advisor.get_dry_run_summary(recommendations), highlight=False)


if __name__ == "__main__":
    app()
