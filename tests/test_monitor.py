"""Tests for real-time online monitoring (OnlineMonitor)."""

import pandas as pd

from devwaste.collectors.docker_collector import ContainerMetric
from devwaste.models.monitor import OnlineMonitor


def make_metric(
    cid: str = "c1",
    name: str = "web",
    cpu: float = 35.0,
    mem: float = 45.0,
    rx: float = 10000.0,
    tx: float = 5000.0,
    ts: str = "2026-10-24T12:00:00Z",
) -> ContainerMetric:
    return ContainerMetric(
        timestamp=ts,
        container_id=cid,
        container_name=name,
        state="running",
        cpu_percent=cpu,
        memory_usage=100 * 1024 * 1024,
        memory_limit=1024 * 1024 * 1024,
        memory_percent=mem,
        network_rx_bytes=rx,
        network_tx_bytes=tx,
        block_read_bytes=500.0,
        block_write_bytes=300.0,
        restart_count=0,
    )


def test_online_monitor_warming_and_ready(trained_model):
    monitor = OnlineMonitor(model_dir=trained_model)
    cid = "c1"

    # Observation 1: first observation, no delta yet
    res1 = monitor.process_metric(make_metric(cid=cid, rx=10000, tx=5000))
    assert res1["status"] == "warming"
    assert res1["samples"] == 0

    # Feed enough observations to complete window
    res = None
    for i in range(monitor.window_size):
        res = monitor.process_metric(
            make_metric(
                cid=cid,
                rx=10000 + (i + 1) * 1000,
                tx=5000 + (i + 1) * 500,
            )
        )

    assert res is not None
    assert res["status"] == "ready"
    assert "score" in res
    assert "threshold" in res
    assert "top_feature" in res
    assert "is_anomalous" in res
    assert res["container_id"] == cid


def test_online_monitor_flags_spike(trained_model, cpu_spike_csv):
    df = pd.read_csv(cpu_spike_csv)
    monitor = OnlineMonitor(model_dir=trained_model)
    cid = df["container_id"].iloc[0]

    rx_cum, tx_cum, r_cum, w_cum = 0.0, 0.0, 0.0, 0.0
    results = []
    for _, row in df.iterrows():
        rx_cum += float(row["network_rx_bytes"])
        tx_cum += float(row["network_tx_bytes"])
        r_cum += float(row["block_read_bytes"])
        w_cum += float(row["block_write_bytes"])
        metric = ContainerMetric(
            timestamp=str(row["timestamp"]),
            container_id=cid,
            container_name="test-container",
            state="running",
            cpu_percent=float(row["cpu_percent"]),
            memory_usage=float(row["memory_usage"]),
            memory_limit=float(row["memory_limit"]),
            memory_percent=float(row["memory_percent"]),
            network_rx_bytes=rx_cum,
            network_tx_bytes=tx_cum,
            block_read_bytes=r_cum,
            block_write_bytes=w_cum,
            restart_count=0,
        )
        res = monitor.process_metric(metric)
        if res.get("status") == "ready":
            results.append(res)

    anomalous = [r for r in results if r["is_anomalous"]]
    assert len(anomalous) > 0
    drivers = [r["top_feature"] for r in anomalous]
    assert "cpu_percent" in drivers


def test_online_monitor_persistence_filter(trained_model, normal_csv):
    df = pd.read_csv(normal_csv)
    monitor = OnlineMonitor(model_dir=trained_model, consecutive=2)
    cid = "c1"

    rx_cum, tx_cum, r_cum, w_cum = 0.0, 0.0, 0.0, 0.0
    # Warm up using normal samples (first 20 rows of normal data)
    for _, row in df.iloc[:20].iterrows():
        rx_cum += float(row["network_rx_bytes"])
        tx_cum += float(row["network_tx_bytes"])
        r_cum += float(row["block_read_bytes"])
        w_cum += float(row["block_write_bytes"])
        metric = ContainerMetric(
            timestamp=str(row["timestamp"]),
            container_id=cid,
            container_name="test-container",
            state="running",
            cpu_percent=float(row["cpu_percent"]),
            memory_usage=float(row["memory_usage"]),
            memory_limit=float(row["memory_limit"]),
            memory_percent=float(row["memory_percent"]),
            network_rx_bytes=rx_cum,
            network_tx_bytes=tx_cum,
            block_read_bytes=r_cum,
            block_write_bytes=w_cum,
            restart_count=0,
        )
        monitor.process_metric(metric)

    # Streak should be reset / zero on normal data
    assert monitor.streak[cid] == 0

    # Inject first spike
    rx_cum += 10000
    res1 = monitor.process_metric(
        ContainerMetric(
            timestamp="2026-10-24T12:00:21Z",
            container_id=cid,
            container_name="test-container",
            state="running",
            cpu_percent=99.0,
            memory_usage=100 * 1024 * 1024,
            memory_limit=1024 * 1024 * 1024,
            memory_percent=45.0,
            network_rx_bytes=rx_cum,
            network_tx_bytes=tx_cum,
            block_read_bytes=r_cum,
            block_write_bytes=w_cum,
            restart_count=0,
        )
    )
    assert res1["raw_anomalous"] is True
    assert res1["streak"] == 1
    assert res1["is_anomalous"] is False

    # Inject second consecutive spike
    rx_cum += 10000
    res2 = monitor.process_metric(
        ContainerMetric(
            timestamp="2026-10-24T12:00:22Z",
            container_id=cid,
            container_name="test-container",
            state="running",
            cpu_percent=99.0,
            memory_usage=100 * 1024 * 1024,
            memory_limit=1024 * 1024 * 1024,
            memory_percent=45.0,
            network_rx_bytes=rx_cum,
            network_tx_bytes=tx_cum,
            block_read_bytes=r_cum,
            block_write_bytes=w_cum,
            restart_count=0,
        )
    )
    assert res2["streak"] == 2
    assert res2["is_anomalous"] is True
