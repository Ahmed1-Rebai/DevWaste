"""Self-contained offline HTML visual reports for DevWaste AI evaluation results."""

from pathlib import Path
from typing import Any


def _fmt(val: Any, digits: int = 3) -> str:
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.{digits}f}"
    return str(val)


def generate_svg_score_chart(
    top_windows: list[dict[str, Any]],
    threshold: float,
    width: int = 800,
    height: int = 220,
) -> str:
    """Generate a clean SVG chart showing top anomalous window scores and threshold."""
    if not top_windows:
        return "<p class='dim'>No window score data available for charting.</p>"

    scores = [float(w.get("score", 0.0)) for w in top_windows]
    max_score = max(scores + [threshold * 1.2, 1e-3])
    pad_left = 60
    pad_right = 30
    pad_top = 25
    pad_bottom = 40
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    n = len(scores)
    step = plot_w / max(n - 1, 1)

    points = []
    circles = []
    for i, s in enumerate(scores):
        x = pad_left + i * step
        y = pad_top + plot_h - (s / max_score) * plot_h
        points.append(f"{x:.1f},{y:.1f}")
        color = "#e53e3e" if s > threshold else "#3182ce"
        circles.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}' "
            f"stroke='#ffffff' stroke-width='1.5'><title>Window {i}: {s:.4f}</title></circle>"
        )

    polyline = " ".join(points)
    thresh_y = pad_top + plot_h - (threshold / max_score) * plot_h

    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg" xmlns="http://www.w3.org/2000/svg">
      <!-- Background grid lines -->
      <line x1="{pad_left}" y1="{pad_top}" x2="{width - pad_right}" y2="{pad_top}" stroke="#2d3748" stroke-dasharray="3,3"/>
      <line x1="{pad_left}" y1="{pad_top + plot_h / 2}" x2="{width - pad_right}" y2="{pad_top + plot_h / 2}" stroke="#2d3748" stroke-dasharray="3,3"/>
      <line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" y2="{pad_top + plot_h}" stroke="#4a5568" stroke-width="1.5"/>

      <!-- Threshold Line -->
      <line x1="{pad_left}" y1="{thresh_y:.1f}" x2="{width - pad_right}" y2="{thresh_y:.1f}" stroke="#ed8936" stroke-width="2" stroke-dasharray="5,5"/>
      <text x="{width - pad_right - 5}" y="{thresh_y - 6:.1f}" fill="#ed8936" font-size="11" text-anchor="end" font-weight="600">Threshold ({threshold:.3f})</text>

      <!-- Score Line -->
      <polyline fill="none" stroke="#63b3ed" stroke-width="2.5" stroke-linecap="round" points="{polyline}"/>
      {"".join(circles)}

      <!-- Y Axis Labels -->
      <text x="{pad_left - 10}" y="{pad_top + 4}" fill="#a0aec0" font-size="11" text-anchor="end">{max_score:.2f}</text>
      <text x="{pad_left - 10}" y="{pad_top + plot_h + 4}" fill="#a0aec0" font-size="11" text-anchor="end">0.00</text>
      <text x="{pad_left - 10}" y="{pad_top + plot_h / 2 + 4}" fill="#a0aec0" font-size="11" text-anchor="end">{max_score / 2:.2f}</text>

      <!-- X Axis Label -->
      <text x="{pad_left + plot_w / 2}" y="{height - 10}" fill="#a0aec0" font-size="11" text-anchor="middle">Top Windows (ranked by error score)</text>
    </svg>
    """


def generate_feature_distribution_bars(
    flagged_by_feature: dict[str, int],
) -> str:
    """Generate visual progress bars for feature attribution breakdown."""
    if not flagged_by_feature:
        return "<p class='dim'>No flagged windows to attribute.</p>"

    total = max(sum(flagged_by_feature.values()), 1)
    rows = []
    colors = ["#4299e1", "#ed8936", "#48bb78", "#9f7aea", "#ed64a6", "#38b2ac"]

    for idx, (feature, count) in enumerate(flagged_by_feature.items()):
        pct = (count / total) * 100
        color = colors[idx % len(colors)]
        rows.append(
            f"""
        <div class="attr-row">
            <div class="attr-label"><span class="badge" style="background:{color}">{feature}</span></div>
            <div class="attr-bar-container">
                <div class="attr-bar" style="width: {pct:.1f}%; background-color: {color};"></div>
            </div>
            <div class="attr-value">{count} ({pct:.1f}%)</div>
        </div>
        """
        )
    return "\n".join(rows)


def generate_html_report(results: dict[str, Any]) -> str:
    """
    Render self-contained HTML visual report from evaluation results.

    Args:
        results: Dictionary returned by evaluate_model

    Returns:
        Complete HTML document string
    """
    title = (
        f"DevWaste AI Diagnostic Report - {Path(results.get('data_path', 'Data')).name}"
    )
    threshold = float(results.get("threshold", 0.0))
    n_windows = results.get("n_windows", 0)
    n_flagged = results.get("n_flagged", 0)
    flagged_fraction = results.get("flagged_fraction", 0.0)
    inference_ms = results.get("inference_ms_per_window", 0.0)
    labelled = results.get("labelled", False)
    flagged_by_feature = results.get("flagged_by_feature", {})
    top_windows = results.get("top_windows", [])

    svg_chart = generate_svg_score_chart(top_windows, threshold)
    attr_bars = generate_feature_distribution_bars(flagged_by_feature)

    # Labelled metrics table
    metrics_section = ""
    if labelled and "model_metrics" in results:
        m = results["model_metrics"]
        baselines = results.get("baseline_metrics", {})
        rows = [
            f"""<tr>
                <td><strong>LSTM Autoencoder</strong></td>
                <td>{_fmt(m.get("precision"))}</td>
                <td>{_fmt(m.get("recall"))}</td>
                <td><strong>{_fmt(m.get("f1_score"))}</strong></td>
                <td>{_fmt(m.get("false_positive_rate"))}</td>
                <td>{_fmt(m.get("auc"))}</td>
            </tr>"""
        ]
        for b_name, b_m in baselines.items():
            rows.append(
                f"""<tr>
                <td>Baseline: {b_name}</td>
                <td>{_fmt(b_m.get("precision"))}</td>
                <td>{_fmt(b_m.get("recall"))}</td>
                <td>{_fmt(b_m.get("f1_score"))}</td>
                <td>{_fmt(b_m.get("false_positive_rate"))}</td>
                <td>{_fmt(b_m.get("auc"))}</td>
            </tr>"""
            )

        metrics_section = f"""
        <div class="card">
            <h2>Detection Performance vs Baselines</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Detector</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>False Positive Rate</th>
                        <th>ROC-AUC</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    # Top windows table
    top_windows_rows = []
    for w in top_windows:
        status_badge = (
            "<span class='badge badge-red'>ANOMALOUS</span>"
            if w.get("flagged")
            else "<span class='badge badge-blue'>NORMAL</span>"
        )
        c_name = w.get("container_name", w.get("container_id", "container"))
        top_windows_rows.append(
            f"""<tr>
            <td>{c_name}</td>
            <td>{w.get("start_timestamp", w.get("start_sample", "n/a"))}</td>
            <td><strong>{_fmt(w.get("score"), 4)}</strong></td>
            <td>{status_badge}</td>
            <td><code>{w.get("top_feature", "n/a")}</code></td>
        </tr>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-red: #f87171;
            --accent-green: #4ade80;
            --accent-yellow: #fbbf24;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            padding: 30px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 25px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
        }}
        .header h1 {{ font-size: 1.75rem; color: var(--accent-blue); }}
        .header p {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .stat-label {{ font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 5px; color: var(--text-primary); }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
        }}
        .card h2 {{ font-size: 1.2rem; margin-bottom: 15px; color: var(--text-primary); border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }}
        .chart-svg {{ width: 100%; height: auto; display: block; }}
        .table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        .table th, .table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border-color); }}
        .table th {{ background: #0f172a; color: var(--text-secondary); font-weight: 600; }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            color: #ffffff;
        }}
        .badge-red {{ background: #e53e3e; }}
        .badge-blue {{ background: #3182ce; }}
        .attr-row {{ display: flex; align-items: center; margin-bottom: 10px; font-size: 0.85rem; }}
        .attr-label {{ width: 180px; }}
        .attr-bar-container {{ flex: 1; background: #334155; border-radius: 4px; height: 16px; margin: 0 12px; overflow: hidden; }}
        .attr-bar {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .attr-value {{ width: 100px; text-align: right; color: var(--text-secondary); font-family: monospace; }}
        .footer {{ font-size: 0.8rem; color: var(--text-secondary); text-align: center; margin-top: 30px; }}
        .dim {{ color: var(--text-secondary); font-style: italic; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DevWaste AI - Anomaly Evaluation Report</h1>
        <p>Data Source: <code>{results.get("data_path", "unknown")}</code> | Model: <code>{results.get("model_dir", "unknown")}</code></p>
    </div>

    <div class="grid">
        <div class="stat-card">
            <div class="stat-label">Scored Windows</div>
            <div class="stat-value">{n_windows}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Flagged Anomalies</div>
            <div class="stat-value" style="color: {"var(--accent-red)" if n_flagged > 0 else "var(--accent-green)"}">{n_flagged}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Flagged Fraction</div>
            <div class="stat-value">{flagged_fraction:.1%}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Reconstruction Threshold</div>
            <div class="stat-value">{threshold:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Inference Latency</div>
            <div class="stat-value">{inference_ms:.2f} ms</div>
        </div>
    </div>

    <div class="card">
        <h2>Reconstruction Error Curve & Threshold</h2>
        {svg_chart}
    </div>

    {metrics_section}

    <div class="card">
        <h2>Flagged Anomaly Feature Breakdown</h2>
        {attr_bars}
    </div>

    <div class="card">
        <h2>Highest-Scoring Incident Windows</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Container / Series</th>
                    <th>Timestamp / Window</th>
                    <th>Error Score</th>
                    <th>Status</th>
                    <th>Primary Feature Driver</th>
                </tr>
            </thead>
            <tbody>
                {"".join(top_windows_rows)}
            </tbody>
        </table>
    </div>

    <div class="footer">
        Generated by DevWaste AI (Diagnostic Only &bull; Local PyTorch Deep Learning)
    </div>
</body>
</html>
"""


def write_html_report(results: dict[str, Any], output_path: Path):
    """Write visual HTML report to the specified file path."""
    html_content = generate_html_report(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
