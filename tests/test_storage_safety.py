"""Tests for SQLite storage and cleanup safety."""

from unittest.mock import Mock

from devwaste.collectors.docker_collector import ContainerMetric
from devwaste.safety.cleanup import CleanupAdvisor
from devwaste.storage.sqlite_store import SQLiteStore


def make_metric(
    ts: str, container_id: str = "c1", cpu: float = 10.0
) -> ContainerMetric:
    return ContainerMetric(
        timestamp=ts,
        container_id=container_id,
        container_name=f"name-{container_id}",
        state="running",
        cpu_percent=cpu,
        memory_usage=1.0,
        memory_limit=2.0,
        memory_percent=50.0,
        network_rx_bytes=0.0,
        network_tx_bytes=0.0,
        block_read_bytes=0.0,
        block_write_bytes=0.0,
        restart_count=0,
    )


class TestSQLiteStore:
    def test_save_and_load(self, tmp_path):
        db = tmp_path / "nested" / "metrics.db"
        with SQLiteStore(db) as store:
            store.save_metric(make_metric("2024-01-01T00:00:02"))
            store.save_metrics([make_metric("2024-01-01T00:00:01", "c2", cpu=20.0)])
            df = store.load_metrics()

        assert db.exists()
        assert len(df) == 2
        assert df["timestamp"].is_monotonic_increasing
        assert df.iloc[0]["cpu_percent"] == 20.0

    def test_filters(self, tmp_path):
        with SQLiteStore(tmp_path / "m.db") as store:
            store.save_metrics(
                [make_metric(f"2024-01-01T00:00:0{i}", f"c{i % 2}") for i in range(6)]
            )

            assert len(store.load_metrics(container_id="c0")) == 3
            assert len(store.load_metrics(start_time="2024-01-01T00:00:04")) == 2
            assert len(store.load_metrics(end_time="2024-01-01T00:00:01")) == 2
            latest = store.get_latest_metrics("c1", n=2)

        assert latest["timestamp"].tolist() == [
            "2024-01-01T00:00:03",
            "2024-01-01T00:00:05",
        ]

    def test_persists_across_connections(self, tmp_path):
        db = tmp_path / "m.db"
        with SQLiteStore(db) as store:
            store.save_metric(make_metric("t1"))
        with SQLiteStore(db) as store:
            assert len(store.load_metrics()) == 1


def make_client(containers=(), images=(), volumes=(), networks=()) -> Mock:
    client = Mock()
    client.containers.list.return_value = list(containers)
    client.images.list.return_value = list(images)
    client.volumes.list.return_value = list(volumes)
    client.networks.list.return_value = list(networks)
    return client


def named(name: str, **attrs) -> Mock:
    obj = Mock()
    obj.name = name
    obj.short_id = f"sha256:{name}"
    obj.attrs = attrs
    return obj


class TestCleanupAdvisor:
    def test_no_recommendations(self):
        advisor = CleanupAdvisor(make_client())

        assert advisor.get_recommendations() == []
        assert "Nothing to recommend" in advisor.get_dry_run_summary()

    def test_recommendations_from_state(self):
        client = make_client(
            containers=[named("old-job")],
            images=[named("img", Size=2048)],
            volumes=[named("pgdata")],
            networks=[named("bridge"), named("proj_default")],
        )
        recs = {r["action"]: r for r in CleanupAdvisor(client).get_recommendations()}

        assert recs["remove_stopped_containers"]["items"] == ["old-job"]
        assert recs["prune_dangling_images"]["reclaimable_bytes"] == 2048
        assert recs["review_unused_volumes"]["risk"] == "high"
        assert recs["prune_unused_networks"]["items"] == ["proj_default"]

    def test_read_only(self):
        """The advisor never calls any mutating Docker API."""
        container = named("old-job")
        image, volume, network = named("img", Size=1), named("v"), named("n")
        client = make_client([container], [image], [volume], [network])

        summary = CleanupAdvisor(client).get_dry_run_summary()

        for obj in (container, image, volume, network):
            obj.remove.assert_not_called()
            obj.stop.assert_not_called()
            obj.kill.assert_not_called()
        for collection in (
            client.containers,
            client.images,
            client.volumes,
            client.networks,
        ):
            collection.prune.assert_not_called()
        assert "No actions were performed" in summary
