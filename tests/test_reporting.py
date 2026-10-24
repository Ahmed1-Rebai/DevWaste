"""Tests for offline visual HTML report generation."""

from devwaste.reporting.html_report import (
    generate_feature_distribution_bars,
    generate_html_report,
    generate_svg_score_chart,
    write_html_report,
)


def test_svg_score_chart():
    top_windows = [
        {"score": 1.5, "flagged": True},
        {"score": 0.8, "flagged": False},
        {"score": 2.2, "flagged": True},
    ]
    svg = generate_svg_score_chart(top_windows, threshold=1.0)
    assert "<svg" in svg
    assert "Threshold" in svg
    assert "<polyline" in svg
    assert "<circle" in svg


def test_feature_distribution_bars():
    bars = generate_feature_distribution_bars({"cpu_percent": 15, "memory_percent": 5})
    assert "cpu_percent" in bars
    assert "memory_percent" in bars
    assert "75.0%" in bars
    assert "25.0%" in bars


def test_full_html_report_rendering(tmp_path):
    results = {
        "data_path": "data/test.csv",
        "model_dir": "artifacts/model",
        "threshold": 1.25,
        "n_windows": 100,
        "n_flagged": 10,
        "flagged_fraction": 0.1,
        "inference_ms_per_window": 0.05,
        "labelled": True,
        "model_metrics": {
            "precision": 0.9,
            "recall": 0.85,
            "f1_score": 0.874,
            "false_positive_rate": 0.02,
            "auc": 0.98,
        },
        "baseline_metrics": {
            "zscore": {
                "precision": 0.8,
                "recall": 0.8,
                "f1_score": 0.8,
                "false_positive_rate": 0.05,
                "auc": 0.92,
            }
        },
        "flagged_by_feature": {"cpu_percent": 8, "network_rx_bytes": 2},
        "top_windows": [
            {
                "container_name": "web-1",
                "start_timestamp": "2026-10-24T12:00:00",
                "score": 3.45,
                "flagged": True,
                "top_feature": "cpu_percent",
            }
        ],
    }

    out_file = tmp_path / "report.html"
    raw_html = generate_html_report(results)
    assert "<!DOCTYPE html>" in raw_html

    write_html_report(results, out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "DevWaste AI - Anomaly Evaluation Report" in content
    assert "web-1" in content
    assert "3.4500" in content
    assert "0.874" in content
