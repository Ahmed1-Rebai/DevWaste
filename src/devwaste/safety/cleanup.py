"""Safety module for cleanup recommendations.

IMPORTANT: Diagnostic only. This module inspects Docker state through
read-only API calls (list/inspect) and suggests commands for a human to
review. It never stops, removes, or prunes anything, and it never
recommends action based on anomaly scores.
"""

from typing import Any

PREDEFINED_NETWORKS = {"bridge", "host", "none"}
MAX_LISTED_ITEMS = 10


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class CleanupAdvisor:
    """Provides read-only, diagnostic cleanup recommendations."""

    def __init__(self, client: Any):
        """
        Initialize cleanup advisor.

        Args:
            client: Docker SDK client (``docker.from_env()``) or a test double
        """
        self.client = client

    def _stopped_containers(self) -> dict[str, Any] | None:
        containers = self.client.containers.list(
            all=True, filters={"status": ["exited", "created", "dead"]}
        )
        if not containers:
            return None
        return {
            "action": "remove_stopped_containers",
            "target": "stopped containers",
            "count": len(containers),
            "items": [c.name for c in containers[:MAX_LISTED_ITEMS]],
            "reclaimable_bytes": None,
            "reason": "Stopped containers keep their writable layer on disk",
            "risk": "medium",
            "caution": "Their filesystem changes and logs are lost on removal",
            "suggested_command": "docker container prune",
        }

    def _dangling_images(self) -> dict[str, Any] | None:
        images = self.client.images.list(filters={"dangling": True})
        if not images:
            return None
        size = sum(img.attrs.get("Size", 0) or 0 for img in images)
        return {
            "action": "prune_dangling_images",
            "target": "dangling images",
            "count": len(images),
            "items": [img.short_id for img in images[:MAX_LISTED_ITEMS]],
            "reclaimable_bytes": size,
            "reason": "Untagged image layers not referenced by any tag",
            "risk": "low",
            "caution": "Size may overlap with layers shared by other images",
            "suggested_command": "docker image prune",
        }

    def _unused_volumes(self) -> dict[str, Any] | None:
        volumes = self.client.volumes.list(filters={"dangling": True})
        if not volumes:
            return None
        return {
            "action": "review_unused_volumes",
            "target": "volumes not used by any container",
            "count": len(volumes),
            "items": [v.name for v in volumes[:MAX_LISTED_ITEMS]],
            "reclaimable_bytes": None,
            "reason": "Volumes persist after their containers are removed",
            "risk": "high",
            "caution": "Volumes may hold databases or other persistent data. "
            "Inspect each one before removing it.",
            "suggested_command": "docker volume rm <name>  # after inspection",
        }

    def _unused_networks(self) -> dict[str, Any] | None:
        networks = [
            n
            for n in self.client.networks.list(filters={"dangling": True})
            if n.name not in PREDEFINED_NETWORKS
        ]
        if not networks:
            return None
        return {
            "action": "prune_unused_networks",
            "target": "networks without containers",
            "count": len(networks),
            "items": [n.name for n in networks[:MAX_LISTED_ITEMS]],
            "reclaimable_bytes": None,
            "reason": "Unused user-defined networks clutter configuration",
            "risk": "low",
            "caution": "Compose projects recreate their networks on next start",
            "suggested_command": "docker network prune",
        }

    def get_recommendations(self) -> list[dict[str, Any]]:
        """
        Inspect Docker state and return cleanup recommendations.

        Returns:
            List of recommendation dictionaries (empty if nothing to suggest)
        """
        checks = (
            self._stopped_containers,
            self._dangling_images,
            self._unused_networks,
            self._unused_volumes,
        )
        return [rec for check in checks if (rec := check()) is not None]

    def get_dry_run_summary(
        self, recommendations: list[dict[str, Any]] | None = None
    ) -> str:
        """
        Get a human-readable summary of recommendations without executing anything.

        Returns:
            Summary string
        """
        if recommendations is None:
            recommendations = self.get_recommendations()

        lines = ["CLEANUP DRY RUN SUMMARY", "=" * 40, ""]
        if not recommendations:
            lines.append(
                "Nothing to recommend: no stopped containers, dangling images,"
            )
            lines.append("unused networks, or unused volumes found.")
            lines.append("")

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec['action']} ({rec['count']} {rec['target']})")
            if rec["items"]:
                more = rec["count"] - len(rec["items"])
                suffix = f" (+{more} more)" if more > 0 else ""
                lines.append(f"   Items: {', '.join(rec['items'])}{suffix}")
            if rec["reclaimable_bytes"]:
                lines.append(
                    f"   Approx. size: {_human_bytes(rec['reclaimable_bytes'])}"
                )
            lines.append(f"   Reason: {rec['reason']}")
            lines.append(f"   Risk: {rec['risk']} - {rec['caution']}")
            lines.append(f"   Review, then run manually: {rec['suggested_command']}")
            lines.append("")

        lines.append("=" * 40)
        lines.append("NOTE: No actions were performed (dry run mode).")
        lines.append("      DevWaste never removes resources automatically, and")
        lines.append(
            "      anomaly scores alone are never a reason to remove anything."
        )

        return "\n".join(lines)
