"""Tests for Docker metrics collector (Docker SDK fully mocked)."""

from unittest.mock import Mock, patch

import pytest

from devwaste.collectors.docker_collector import DockerCollector, DockerException


def make_container(status: str = "running") -> Mock:
    container = Mock()
    container.id = "abc123def456789"
    container.name = "web"
    container.status = status
    container.attrs = {"RestartCount": 2, "Config": {"Image": "nginx:latest"}}
    return container


STATS = {
    "precpu_stats": {
        "cpu_usage": {"total_usage": 1_000_000_000},
        "system_cpu_usage": 20_000_000_000,
    },
    "cpu_stats": {
        "cpu_usage": {"total_usage": 1_500_000_000},
        "system_cpu_usage": 22_000_000_000,
        "online_cpus": 4,
    },
    "memory_stats": {
        "usage": 600,
        "limit": 1000,
        "stats": {"inactive_file": 100},
    },
    "networks": {"eth0": {"rx_bytes": 10, "tx_bytes": 20}},
    "blkio_stats": {
        "io_service_bytes_recursive": [
            {"op": "read", "value": 5},
            {"op": "write", "value": 7},
        ]
    },
}


class TestDockerCollector:
    """Test DockerCollector class."""

    @patch("devwaste.collectors.docker_collector.docker")
    def test_initialization_success(self, mock_docker):
        """Test successful client initialization."""
        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client

        collector = DockerCollector()

        assert collector.client == mock_client
        mock_client.ping.assert_called_once()

    @patch("devwaste.collectors.docker_collector.docker")
    def test_initialization_failure(self, mock_docker):
        """Test initialization when Docker is unavailable."""
        mock_docker.from_env.side_effect = DockerException("Cannot connect")

        with pytest.raises(RuntimeError, match="Cannot connect to Docker daemon"):
            DockerCollector()

    def test_injected_client(self):
        """A provided client is used without contacting Docker."""
        client = Mock()
        assert DockerCollector(client=client).client is client
        client.ping.assert_not_called()

    def test_cpu_percent_calculation(self):
        """CPU percentage matches the docker stats formula."""
        collector = DockerCollector(client=Mock())
        prev = {"cpu_stats": STATS["precpu_stats"]}

        cpu_percent = collector._calculate_cpu_percent(prev, STATS)

        # 0.5e9 / 2e9 * 4 CPUs * 100
        assert cpu_percent == pytest.approx(100.0)

    def test_cpu_percent_zero_delta(self):
        """Test CPU percentage when delta is zero."""
        collector = DockerCollector(client=Mock())
        stats = {"cpu_stats": STATS["precpu_stats"]}

        assert collector._calculate_cpu_percent(stats, stats) == 0.0

    def test_cpu_percent_exception_handling(self):
        """Test CPU percentage handles missing data gracefully."""
        collector = DockerCollector(client=Mock())

        assert collector._calculate_cpu_percent({}, {}) == 0.0

    def test_memory_stats_subtracts_cache(self):
        """Page cache is excluded from usage, like docker stats."""
        collector = DockerCollector(client=Mock())

        usage, limit, percent = collector._get_memory_stats(STATS)

        assert usage == 500
        assert limit == 1000
        assert percent == 50.0

    def test_memory_stats_zero_limit(self):
        """Test memory statistics with zero limit."""
        collector = DockerCollector(client=Mock())

        _, _, percent = collector._get_memory_stats(
            {"memory_stats": {"usage": 2, "limit": 0}}
        )

        assert percent == 0.0

    def test_network_stats_extraction(self):
        """Test network statistics summed over interfaces."""
        collector = DockerCollector(client=Mock())
        stats = {
            "networks": {
                "eth0": {"rx_bytes": 1000, "tx_bytes": 500},
                "eth1": {"rx_bytes": 2000, "tx_bytes": 1000},
            }
        }

        assert collector._get_network_stats(stats) == (3000, 1500)

    def test_network_stats_missing(self):
        """Containers with no network (e.g. --network none) report zero."""
        collector = DockerCollector(client=Mock())

        assert collector._get_network_stats({"networks": None}) == (0, 0)

    def test_block_stats_legacy_format(self):
        """Older API format with 'bytes' keys and mixed-case ops."""
        collector = DockerCollector(client=Mock())
        stats = {
            "blkio_stats": {
                "io_service_bytes": [
                    {"op": "Read", "bytes": 1000},
                    {"op": "Write", "bytes": 500},
                    {"op": "read", "bytes": 200},
                    {"op": "write", "bytes": 300},
                ]
            }
        }

        assert collector._get_block_stats(stats) == (1200, 800)

    def test_metric_from_stats_uses_precpu(self):
        """A single stats response yields a full metric with non-zero CPU."""
        collector = DockerCollector(client=Mock())

        metric = collector.metric_from_stats(make_container(), STATS, timestamp="t0")

        assert metric.cpu_percent == pytest.approx(100.0)
        assert metric.memory_percent == 50.0
        assert (metric.network_rx_bytes, metric.network_tx_bytes) == (10, 20)
        assert (metric.block_read_bytes, metric.block_write_bytes) == (5, 7)
        assert metric.restart_count == 2
        assert metric.container_name == "web"

    def test_collect_metrics_handles_errors(self):
        """A container failing mid-collection is skipped, not fatal."""
        container = make_container()
        container.stats.side_effect = Exception("gone")
        collector = DockerCollector(client=Mock())

        assert collector.collect_metrics(container) is None

    def test_collect_all_metrics(self):
        """Collect from every listed container."""
        ok, broken = make_container(), make_container()
        ok.stats.return_value = STATS
        broken.stats.side_effect = Exception("gone")
        client = Mock()
        client.containers.list.return_value = [ok, broken]

        metrics = DockerCollector(client=client).collect_all_metrics()

        assert len(metrics) == 1
        client.containers.list.assert_called_once_with(all=False)


@pytest.mark.integration
def test_live_docker_scan():
    """Requires a running Docker daemon: pytest -m integration."""
    collector = DockerCollector()
    for metric in collector.collect_all_metrics():
        assert metric.cpu_percent >= 0
