"""Docker metrics collector.

Read-only: this module only lists containers and reads their stats. It
never reads environment variables, logs, or filesystem contents.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

try:
    import docker
    from docker.errors import DockerException

    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False
    DockerException = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


@dataclass
class ContainerMetric:
    """
    Single container metric observation.

    Network and block I/O values are cumulative counters since container
    start, as reported by Docker. Convert them with
    :func:`devwaste.data.features.cumulative_to_deltas` before modelling.
    """

    timestamp: str
    container_id: str
    container_name: str
    state: str
    cpu_percent: float
    memory_usage: float  # bytes
    memory_limit: float  # bytes
    memory_percent: float
    network_rx_bytes: float
    network_tx_bytes: float
    block_read_bytes: float
    block_write_bytes: float
    restart_count: int


class DockerCollector:
    """Collect metrics from Docker containers."""

    def __init__(self, client: Any | None = None):
        """
        Initialize Docker client.

        Args:
            client: Optional pre-built Docker client (useful for testing)

        Raises:
            ImportError: If the Docker SDK is not installed
            RuntimeError: If the Docker daemon is unreachable
        """
        if client is not None:
            self.client = client
            return

        if not HAS_DOCKER:
            raise ImportError(
                "Docker SDK not installed. Install with: pip install docker"
            )

        try:
            self.client = docker.from_env()
            self.client.ping()
        except DockerException as e:
            raise RuntimeError(f"Cannot connect to Docker daemon: {e}") from e

    def list_containers(self, running_only: bool = True) -> list[Any]:
        """List containers."""
        return self.client.containers.list(all=not running_only)

    def _calculate_cpu_percent(
        self,
        prev_stats: dict[str, Any],
        curr_stats: dict[str, Any],
    ) -> float:
        """
        Calculate CPU percentage from stats deltas (same formula as ``docker stats``).

        ``prev_stats``/``curr_stats`` are ``cpu_stats``-bearing dicts; a
        single Docker stats response provides both via ``precpu_stats`` and
        ``cpu_stats``.
        """
        try:
            prev_cpu = prev_stats["cpu_stats"]["cpu_usage"]["total_usage"]
            curr_cpu = curr_stats["cpu_stats"]["cpu_usage"]["total_usage"]

            prev_sys = prev_stats["cpu_stats"]["system_cpu_usage"]
            curr_sys = curr_stats["cpu_stats"]["system_cpu_usage"]

            cpu_delta = curr_cpu - prev_cpu
            sys_delta = curr_sys - prev_sys

            if sys_delta > 0 and cpu_delta > 0:
                online_cpus = curr_stats["cpu_stats"].get("online_cpus")
                if not online_cpus:
                    online_cpus = len(
                        curr_stats["cpu_stats"]["cpu_usage"].get("percpu_usage") or [1]
                    )

                cpu_percent = (cpu_delta / sys_delta) * online_cpus * 100.0
                return min(cpu_percent, online_cpus * 100.0)
            return 0.0
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0

    def _get_memory_stats(self, stats: dict[str, Any]) -> tuple[float, float, float]:
        """
        Extract memory statistics.

        Page cache is subtracted from usage, matching ``docker stats``
        (``inactive_file`` on cgroup v2, ``total_inactive_file`` on v1).
        """
        try:
            memory_stats = stats["memory_stats"]
            memory_usage = memory_stats["usage"]
            details = memory_stats.get("stats") or {}
            cache = details.get("inactive_file", details.get("total_inactive_file", 0))
            if 0 < cache < memory_usage:
                memory_usage -= cache
            memory_limit = memory_stats.get("limit", 0)
            memory_percent = (
                (memory_usage / memory_limit * 100.0) if memory_limit > 0 else 0.0
            )
            return memory_usage, memory_limit, memory_percent
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0, 0.0, 0.0

    def _get_network_stats(self, stats: dict[str, Any]) -> tuple[float, float]:
        """Extract cumulative network statistics summed over interfaces."""
        try:
            networks = stats.get("networks") or {}
            rx_bytes = sum(net.get("rx_bytes", 0) for net in networks.values())
            tx_bytes = sum(net.get("tx_bytes", 0) for net in networks.values())
            return rx_bytes, tx_bytes
        except (KeyError, TypeError, AttributeError):
            return 0.0, 0.0

    def _get_block_stats(self, stats: dict[str, Any]) -> tuple[float, float]:
        """Extract cumulative block I/O statistics."""
        try:
            entries = (stats.get("blkio_stats") or {}).get("io_service_bytes_recursive")
            if not entries:
                entries = (stats.get("blkio_stats") or {}).get("io_service_bytes") or []
            read_bytes = sum(
                blk.get("value", blk.get("bytes", 0))
                for blk in entries
                if blk.get("op", "").lower() == "read"
            )
            write_bytes = sum(
                blk.get("value", blk.get("bytes", 0))
                for blk in entries
                if blk.get("op", "").lower() == "write"
            )
            return read_bytes, write_bytes
        except (KeyError, TypeError, AttributeError):
            return 0.0, 0.0

    def metric_from_stats(
        self, container: Any, stats: dict[str, Any], timestamp: str | None = None
    ) -> ContainerMetric:
        """Build a :class:`ContainerMetric` from one Docker stats response."""
        cpu_percent = self._calculate_cpu_percent(
            {"cpu_stats": stats.get("precpu_stats") or {}}, stats
        )
        memory_usage, memory_limit, memory_percent = self._get_memory_stats(stats)
        network_rx, network_tx = self._get_network_stats(stats)
        block_read, block_write = self._get_block_stats(stats)

        return ContainerMetric(
            timestamp=timestamp or datetime.now(UTC).isoformat(),
            container_id=container.id,
            container_name=container.name,
            state=container.status,
            cpu_percent=float(cpu_percent),
            memory_usage=float(memory_usage),
            memory_limit=float(memory_limit),
            memory_percent=float(memory_percent),
            network_rx_bytes=float(network_rx),
            network_tx_bytes=float(network_tx),
            block_read_bytes=float(block_read),
            block_write_bytes=float(block_write),
            restart_count=int(container.attrs.get("RestartCount", 0) or 0),
        )

    def collect_metrics(self, container: Any) -> ContainerMetric | None:
        """
        Collect metrics from a single container.

        Uses a non-streaming stats call; Docker samples twice (~1s apart) so
        ``precpu_stats`` is populated for the CPU calculation.
        """
        try:
            container.reload()
            stats = container.stats(stream=False)
            return self.metric_from_stats(container, stats)
        except Exception as e:  # container may disappear between list and stats
            logger.warning(
                "Error collecting metrics from container %s: %s", container.id, e
            )
            return None

    def collect_all_metrics(
        self, running_only: bool = True, max_workers: int = 8
    ) -> list[ContainerMetric]:
        """
        Collect metrics from all containers.

        Stats calls block for ~1-2s each, so they run in a small thread pool.
        Results keep the container listing order.
        """
        containers = self.list_containers(running_only=running_only)
        if not containers:
            return []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(containers))) as pool:
            results = list(pool.map(self.collect_metrics, containers))
        return [m for m in results if m is not None]
