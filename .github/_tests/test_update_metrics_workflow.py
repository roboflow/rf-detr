# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for weekly metrics workflow's repository-writing contract."""

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture(scope="session")
def metrics_workflow(repo_root: Path) -> dict[str, Any]:
    """Parse weekly metrics workflow.

    Examples:
        >>> metrics_workflow  # doctest: +SKIP
        pytest fixture; reads .github/workflows/update-metrics-svg.yml.
    """
    path = repo_root / ".github" / "workflows" / "update-metrics-svg.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def metrics_steps(metrics_workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index update job steps by name.

    Examples:
        >>> metrics_steps  # doctest: +SKIP
        pytest fixture; indexes parsed workflow steps.
    """
    steps = metrics_workflow["jobs"]["update-metrics"]["steps"]
    return {step["name"]: step for step in steps}


class TestUpdateMetricsWorkflow:
    """Tests for scheduled metrics generation and pull-request wiring."""

    def test_schedule_waits_for_completed_pypi_week(self, metrics_workflow: dict[str, Any]) -> None:
        """Schedule must wait until completed Sunday data can be published."""
        triggers = metrics_workflow[True] if True in metrics_workflow else metrics_workflow["on"]

        assert triggers["schedule"] == [{"cron": "0 6 * * 1"}]
        assert "workflow_dispatch" in triggers

    def test_workflow_has_only_required_write_permissions(self, metrics_workflow: dict[str, Any]) -> None:
        """Automation must receive only permissions needed to update its pull request."""
        assert metrics_workflow["permissions"] == {"contents": "write", "pull-requests": "write"}

    def test_open_pr_history_is_restored_from_fixed_branch(
        self,
        metrics_steps: dict[str, dict[str, Any]],
    ) -> None:
        """Open automation pull requests must retain their unmerged SVG checkpoints."""
        restore = metrics_steps["📚 Restore unmerged metrics history"]

        assert restore["env"]["METRICS_BRANCH"] == "automation/update-weekly-metrics"
        assert '--head "$METRICS_BRANCH"' in restore["run"]
        assert "git show FETCH_HEAD:docs/assets/weekly-metrics.svg > docs/assets/weekly-metrics.svg" in restore["run"]

    def test_pull_request_updates_only_metrics_svg(self, metrics_steps: dict[str, dict[str, Any]]) -> None:
        """Pull-request action must write only generated SVG on stable automation branch."""
        create_pull_request = metrics_steps["📨 Create or update metrics pull request"]

        assert create_pull_request["uses"].startswith("peter-evans/create-pull-request@")
        assert create_pull_request["with"]["add-paths"] == "docs/assets/weekly-metrics.svg"
        assert create_pull_request["with"]["base"] == "${{ github.event.repository.default_branch }}"
        assert create_pull_request["with"]["branch"] == "automation/update-weekly-metrics"
        assert create_pull_request["with"]["delete-branch"] is True
