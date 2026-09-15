# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for weekly metrics workflow's repository-writing contract."""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ACTION_PIN = re.compile(r"(?P<action>[\w.-]+/[\w.-]+)@(?P<sha>[0-9a-f]{40})")
RESTORE_STEP = "📚 Restore unmerged metrics history"
TRACKED_SVG = "docs/assets/weekly-metrics.svg"
CHECKED_IN_SVG = "<svg><!-- merged history --></svg>\n"
UNMERGED_SVG = "<svg><!-- unmerged history --></svg>\n"
STUB_LOG_NAME = "commands.log"
requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="restore step is a bash run block")


def _sha_pinned_action(uses: str) -> str | None:
    """Return the owner/repo behind a `uses:` reference pinned to a full commit SHA.

    A tag or branch reference yields `None` instead. Those are mutable, so the action code a
    supply-chain review signed off on can be swapped upstream without any edit landing here.

    Args:
        uses: Value of a workflow step's `uses` key.

    Returns:
        The pinned owner/repo, or `None` when the reference is not a full commit SHA.

    Examples:
        >>> _sha_pinned_action("actions/checkout@" + "0" * 40)
        'actions/checkout'
        >>> _sha_pinned_action("actions/checkout@v6.0.1") is None
        True
    """
    match = ACTION_PIN.fullmatch(uses)
    return match.group("action") if match else None


def _write_stub(directory: Path, name: str, body: str) -> Path:
    """Write an executable shell stub that shadows a real command on PATH.

    Args:
        directory: Directory the stub is written into, to be prepended to PATH.
        name: Command name the stub stands in for.
        body: Shell body, without the shebang line.

    Returns:
        Path of the written stub.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     stub = _write_stub(Path(tmp), "gh", "echo 1")
        ...     (stub.name, os.access(stub, os.X_OK))
        ('gh', True)
    """
    stub = directory / name
    stub.write_text(f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_step(run: str, workspace: Path, stubs: Path, stub_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute a workflow `run` block against a scratch workspace with stubbed commands.

    The block is taken from the parsed workflow rather than copied, so it is the shipped shell that
    runs here. Workflow expressions cannot be evaluated outside a runner and are rejected instead.

    Args:
        run: Body of a step's `run` block.
        workspace: Directory the block runs in, standing in for the runner workspace.
        stubs: Directory of executable stubs, prepended to PATH.
        stub_env: Step environment plus any variables the stubs themselves read.

    Returns:
        The finished `bash` process, with output captured.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _run_step("exit 3", Path(tmp), Path(tmp), {}).returncode
        3
    """
    assert "${{" not in run, "run block reads a workflow expression that only a runner can evaluate"
    script = workspace / "step.sh"
    script.write_text(run, encoding="utf-8")
    env = {**os.environ, **stub_env, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(["bash", str(script)], cwd=workspace, env=env, text=True, capture_output=True, check=False)


def _restore_env(workspace: Path, branch_exists: bool, git_show_fails: bool = False) -> dict[str, str]:
    """Build the environment a restore-step run sees, including the stub controls.

    Args:
        workspace: Directory the step runs in; the stub command log is written beside the step script.
        branch_exists: Whether the `git` stub's `fetch` reports the automation branch as found.
        git_show_fails: Whether the `git` stub rejects `git show` the way a missing path does.

    Returns:
        Environment overlay handed to `_run_step`.

    Examples:
        >>> _restore_env(Path("workspace"), branch_exists=False)["STUB_FETCH_FAILS"]
        '1'
    """
    env = {
        "METRICS_BRANCH": "automation/update-weekly-metrics",
        "STUB_LOG": str(workspace / STUB_LOG_NAME),
        "STUB_UNMERGED_SVG": UNMERGED_SVG,
    }
    if not branch_exists:
        env["STUB_FETCH_FAILS"] = "1"
    if git_show_fails:
        env["STUB_GIT_SHOW_FAILS"] = "1"
    return env


@pytest.fixture
def restore_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """Workspace holding a checked-in SVG, plus a stub recording every `git` call.

    The stub's `fetch` reports the automation branch missing when `STUB_FETCH_FAILS` is set, the way a
    branch that was never pushed (or was deleted after merge) does; its `show` serves `STUB_UNMERGED_SVG`
    and fails like a missing path when `STUB_GIT_SHOW_FAILS` is set.

    Examples:
        >>> restore_sandbox  # doctest: +SKIP
        pytest fixture; builds a workspace and a stub directory under tmp_path.
    """
    workspace = tmp_path / "workspace"
    (workspace / "docs" / "assets").mkdir(parents=True)
    (workspace / TRACKED_SVG).write_text(CHECKED_IN_SVG, encoding="utf-8")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(
        stubs,
        "git",
        'echo "git $*" >> "$STUB_LOG"\n'
        "case $1 in\n"
        "  fetch)\n"
        '    if [ -n "${STUB_FETCH_FAILS:-}" ]; then\n'
        '      echo "fatal: couldn'
        "'"
        't find remote ref $3" >&2\n'
        "      exit 128\n"
        "    fi\n"
        "    ;;\n"
        "  show)\n"
        '    if [ -n "${STUB_GIT_SHOW_FAILS:-}" ]; then\n'
        '      echo "fatal: path does not exist in FETCH_HEAD" >&2\n'
        "      exit 128\n"
        "    fi\n"
        '    printf "%s" "$STUB_UNMERGED_SVG"\n'
        "    ;;\n"
        "esac",
    )
    return workspace, stubs


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

    def test_schedule_waits_for_completed_pypi_week_and_retries_it(self, metrics_workflow: dict[str, Any]) -> None:
        """Schedule must wait for completed Sunday data and retry the same week for three more days.

        PyPI Stats publishes no completion SLA, so a Monday run can fail on data that is not there yet. Every weekday in
        the Monday-to-Thursday window resolves to the same completed week, so the retries recover it; a Monday-only
        schedule would lose that week permanently, because the next Monday sees a non-contiguous history instead.
        """
        triggers = metrics_workflow[True] if True in metrics_workflow else metrics_workflow["on"]

        assert triggers["schedule"] == [{"cron": "0 6 * * 1-4"}]
        assert "workflow_dispatch" in triggers

    def test_workflow_has_only_required_write_permissions(self, metrics_workflow: dict[str, Any]) -> None:
        """Automation must receive only permissions needed to update its pull request."""
        assert metrics_workflow["permissions"] == {"contents": "write", "pull-requests": "write"}

    def test_runs_never_cancel_one_another(self, metrics_workflow: dict[str, Any]) -> None:
        """Overlapping runs must queue behind one another instead of cancelling.

        Each run reads the previous checkpoint out of the committed SVG and writes the next one back. A cancelled run
        can leave the automation branch a week behind, and the run that replaced it would then record a non-contiguous
        week and drop its star delta.
        """
        assert metrics_workflow["concurrency"] == {"group": "weekly-metrics-svg", "cancel-in-progress": False}

    def test_job_cannot_run_unbounded(self, metrics_workflow: dict[str, Any]) -> None:
        """Job must carry an explicit timeout rather than inherit the six-hour default.

        Both upstream APIs are unauthenticated reads with their own 30-second timeouts, so a run that is still alive
        minutes later is stuck, not slow, and holds the serialized queue behind it.
        """
        assert metrics_workflow["jobs"]["update-metrics"]["timeout-minutes"] == 5

    @pytest.mark.parametrize(
        ("step_name", "action"),
        [
            pytest.param("📥 Checkout the repository", "actions/checkout", id="checkout"),
            pytest.param("🐍 Install uv and set Python", "astral-sh/setup-uv", id="setup-uv"),
            pytest.param(
                "📨 Create or update metrics pull request",
                "peter-evans/create-pull-request",
                id="create-pull-request",
            ),
        ],
    )
    def test_third_party_actions_are_pinned_to_commit_shas(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        step_name: str,
        action: str,
    ) -> None:
        """Every third-party action must be pinned to an immutable commit SHA.

        This job holds write access to the repository, so a tag pin would let an upstream retag hand that access to code
        nobody here reviewed.
        """
        assert _sha_pinned_action(metrics_steps[step_name]["uses"]) == action

    def test_checkout_reads_the_default_branch(self, metrics_steps: dict[str, dict[str, Any]]) -> None:
        """Checkout must read the default branch rather than the automation branch.

        A scheduled run checks out whatever ref it is given. Taking the automation branch would stack each week's
        generated SVG on the previous pull request instead of on the merged history.
        """
        assert metrics_steps["📥 Checkout the repository"]["with"]["ref"] == (
            "${{ github.event.repository.default_branch }}"
        )

    def test_metrics_branch_history_is_restored_from_fixed_branch(
        self,
        metrics_steps: dict[str, dict[str, Any]],
    ) -> None:
        """An existing automation branch must hand back its unmerged SVG checkpoints.

        Restoration is keyed on the branch existing, not on whether it currently has an open pull request — a pull
        request can go stale or be closed without the branch being deleted.
        """
        restore = metrics_steps[RESTORE_STEP]

        assert restore["env"]["METRICS_BRANCH"] == "automation/update-weekly-metrics"
        assert 'git fetch --no-tags --depth=1 origin "$METRICS_BRANCH"' in restore["run"]
        assert "FETCH_HEAD:docs/assets/weekly-metrics.svg" in restore["run"]

    def test_restored_svg_lands_through_a_temporary_file(self, metrics_steps: dict[str, dict[str, Any]]) -> None:
        """Restore must never redirect `git show` straight onto the tracked SVG.

        The shell truncates a redirect target before the command on its left runs. Writing onto the tracked path would
        therefore empty the checked-in SVG whenever the metrics branch no longer carries it, and the generator would
        then fail to parse its own checkpoint metadata.
        """
        run = metrics_steps[RESTORE_STEP]["run"]

        assert "> docs/assets/weekly-metrics.svg.tmp" in run
        assert "mv docs/assets/weekly-metrics.svg.tmp docs/assets/weekly-metrics.svg" in run

    def test_pull_request_updates_only_metrics_svg(self, metrics_steps: dict[str, dict[str, Any]]) -> None:
        """Pull-request action must write only generated SVG on stable automation branch."""
        create_pull_request = metrics_steps["📨 Create or update metrics pull request"]

        assert create_pull_request["with"]["add-paths"] == "docs/assets/weekly-metrics.svg"
        assert create_pull_request["with"]["base"] == "${{ github.event.repository.default_branch }}"
        assert create_pull_request["with"]["branch"] == "automation/update-weekly-metrics"
        assert create_pull_request["with"]["delete-branch"] is True


@requires_bash
class TestRestoreUnmergedHistoryStep:
    """Tests that run the restore step's own shell against a stubbed `git`.

    The step is the only part of this workflow that is shell rather than Python, so nothing else in the suite covers
    what it actually does to the working tree. These tests execute the `run` block straight out of the parsed workflow,
    which keeps them honest about the shipped shell.
    """

    def test_existing_branch_hands_back_its_unmerged_svg(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """An existing automation branch must hand its unmerged SVG to the next run.

        The generator reads its previous checkpoint out of the SVG it is about to overwrite. Starting from the merged
        copy while the automation branch still carries unmerged weeks would silently drop them and re-baseline the star
        delta — regardless of whether a pull request for that branch happens to be open right now.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"], workspace, stubs, _restore_env(workspace, branch_exists=True)
        )

        assert result.returncode == 0, result.stderr
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == UNMERGED_SVG

    def test_failed_restore_leaves_the_checked_in_svg_intact(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """A metrics branch no longer carrying the SVG must not empty the checked-in one.

        This is the regression behind the temporary-file staging: a redirect straight onto the tracked
        path truncates it before `git show` reports the missing path, and the generator would then read
        an empty file as a corrupt checkpoint.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"],
            workspace,
            stubs,
            _restore_env(workspace, branch_exists=True, git_show_fails=True),
        )

        assert result.returncode != 0
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG

    def test_failed_restore_reports_a_workflow_error_annotation(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """A failed restore must name itself in the run summary.

        A bare non-zero exit from a compound shell block points at the step, not at the command inside it that failed,
        which is what makes a scheduled failure expensive to diagnose weeks later.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"],
            workspace,
            stubs,
            _restore_env(workspace, branch_exists=True, git_show_fails=True),
        )

        assert "::error::" in result.stdout

    def test_absent_branch_leaves_the_working_tree_alone(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """With no automation branch on the remote the step must rewrite nothing.

        A first-ever run (or one after the automation branch was deleted on merge) has nothing to restore; the checked-
        out default branch already holds the newest checkpoint.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"], workspace, stubs, _restore_env(workspace, branch_exists=False)
        )

        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG
        assert result.returncode == 0, result.stderr

    def test_fetch_is_scoped_to_the_automation_branch(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """The branch lookup must fetch only the fixed automation branch by name.

        An unscoped fetch would pull unrelated refs, which would restore the wrong branch's history over the merged copy
        on nearly every run.
        """
        workspace, stubs = restore_sandbox

        _run_step(metrics_steps[RESTORE_STEP]["run"], workspace, stubs, _restore_env(workspace, branch_exists=True))

        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8")
        assert "fetch --no-tags --depth=1 origin automation/update-weekly-metrics" in logged
