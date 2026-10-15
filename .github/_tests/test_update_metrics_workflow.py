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
STUB_BRANCH_SHA = "deadbeef"
GITHUB_ENV_NAME = "github_env"
STEP_SUMMARY_NAME = "step_summary"
requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="restore step is a bash run block")
requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="the real-git tests drive an actual repository")
COMMIT_STEP = "📤 Commit and push metrics update"
GENERATE_STEP = "📊 Generate weekly metrics SVG"
METRICS_BRANCH = "automation/update-weekly-metrics"


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


def _restore_env(
    workspace: Path,
    branch_exists: bool,
    git_show_fails: bool = False,
    fetch_fails: bool = False,
    ls_remote_status: int | None = None,
) -> dict[str, str]:
    """Build the environment a restore-step run sees, including the stub controls.

    Args:
        workspace: Directory the step runs in; the stub command log and `GITHUB_ENV` file live beside the script.
        branch_exists: Whether the `git` stub's `ls-remote` finds the automation branch (exit 0) or not (exit 2).
        git_show_fails: Whether the `git` stub rejects `git show` the way a missing path does.
        fetch_fails: Whether the `git` stub rejects `git fetch` even though `ls-remote` found the branch.
        ls_remote_status: Explicit `git ls-remote` exit status to simulate, overriding `branch_exists`.

    Returns:
        Environment overlay handed to `_run_step`.

    Examples:
        >>> _restore_env(Path("workspace"), branch_exists=False)["STUB_LS_REMOTE_STATUS"]
        '2'
        >>> _restore_env(Path("workspace"), branch_exists=True, fetch_fails=True)["STUB_FETCH_FAILS"]
        '1'
    """
    env = {
        "METRICS_BRANCH": METRICS_BRANCH,
        "GITHUB_ENV": str(workspace / GITHUB_ENV_NAME),
        "STUB_LOG": str(workspace / STUB_LOG_NAME),
        "STUB_UNMERGED_SVG": UNMERGED_SVG,
        "STUB_LS_REMOTE_STATUS": str(ls_remote_status if ls_remote_status is not None else (0 if branch_exists else 2)),
    }
    if fetch_fails:
        env["STUB_FETCH_FAILS"] = "1"
    if git_show_fails:
        env["STUB_GIT_SHOW_FAILS"] = "1"
    return env


def _commit_env(workspace: Path, branch_exists: bool, svg_changed: bool, push_fails: bool = False) -> dict[str, str]:
    """Build the environment a commit-step run sees, including the stub controls.

    Args:
        workspace: Directory the step runs in; the stub command log and step summary live beside the step script.
        branch_exists: Whether the automation branch existed on the remote when the run started. When it did,
            the restore step has exported its tip as `METRICS_BASE_SHA`; otherwise that variable is unset.
        svg_changed: Whether the `git diff --quiet` guard reports a tracked SVG change.
        push_fails: Whether the `git` stub rejects the publishing call the way a lost lease or a refused ref does.

    Returns:
        Environment overlay handed to `_run_step`.

    Examples:
        >>> env = _commit_env(Path("workspace"), branch_exists=False, svg_changed=False)
        >>> (env["STUB_DIFF_CLEAN"], "METRICS_BASE_SHA" in env, env["DEFAULT_BRANCH"])
        ('1', False, 'main')
        >>> _commit_env(Path("workspace"), branch_exists=True, svg_changed=True)["METRICS_BASE_SHA"]
        'deadbeef'
        >>> _commit_env(Path("workspace"), branch_exists=True, svg_changed=True, push_fails=True)["STUB_PUSH_FAILS"]
        '1'
    """
    env = {
        "METRICS_BRANCH": METRICS_BRANCH,
        "DEFAULT_BRANCH": "main",
        "GITHUB_REPOSITORY": "roboflow/rf-detr",
        "GITHUB_STEP_SUMMARY": str(workspace / STEP_SUMMARY_NAME),
        "STUB_LOG": str(workspace / STUB_LOG_NAME),
    }
    if branch_exists:
        env["METRICS_BASE_SHA"] = STUB_BRANCH_SHA
    if not svg_changed:
        env["STUB_DIFF_CLEAN"] = "1"
    if push_fails:
        env["STUB_PUSH_FAILS"] = "1"
    return env


@pytest.fixture
def restore_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """Workspace holding a checked-in SVG, plus a stub recording every `git` call.

    The stub's `ls-remote` exits with `STUB_LS_REMOTE_STATUS` (2 for a branch that was never pushed or was
    deleted after merge), `fetch` fails when `STUB_FETCH_FAILS` is set, `rev-parse` reports the branch tip, and
    `show` serves `STUB_UNMERGED_SVG` and fails like a missing path when `STUB_GIT_SHOW_FAILS` is set.

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
        "  ls-remote)\n"
        '    exit "${STUB_LS_REMOTE_STATUS:-0}"\n'
        "    ;;\n"
        "  fetch)\n"
        '    if [ -n "${STUB_FETCH_FAILS:-}" ]; then\n'
        '      echo "fatal: unable to fetch $5" >&2\n'
        "      exit 128\n"
        "    fi\n"
        "    ;;\n"
        "  rev-parse)\n"
        f'    echo "{STUB_BRANCH_SHA}"\n'
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


@pytest.fixture
def commit_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """Workspace plus a git stub for exercising the commit-and-push step.

    The stub reports whether the tracked SVG changed through `STUB_DIFF_CLEAN`, rejects `git push` when
    `STUB_PUSH_FAILS` is set, and records every `git` call in `STUB_LOG` so tests can assert how the step tried to
    publish its update.

    Examples:
        >>> commit_sandbox  # doctest: +SKIP
        pytest fixture; builds a workspace and a stub directory under tmp_path.
    """
    workspace = tmp_path / "workspace"
    (workspace / "docs" / "assets").mkdir(parents=True)
    (workspace / TRACKED_SVG).write_text(UNMERGED_SVG, encoding="utf-8")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    _write_stub(
        stubs,
        "git",
        'echo "git $*" >> "$STUB_LOG"\n'
        "case $1 in\n"
        "  diff)\n"
        '    if [ "$2" = "--quiet" ] && [ -n "${STUB_DIFF_CLEAN:-}" ]; then\n'
        "      exit 0\n"
        "    fi\n"
        '    if [ "$2" = "--quiet" ]; then\n'
        "      exit 1\n"
        "    fi\n"
        "    ;;\n"
        "  push)\n"
        '    if [ -n "${STUB_PUSH_FAILS:-}" ]; then\n'
        '      echo "! [rejected] HEAD -> $3 (stale info)" >&2\n'
        "      exit 1\n"
        "    fi\n"
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
    """Tests for the workflow's schedule, permissions, action pins and step contracts."""

    def test_schedule_waits_for_completed_pypi_week_and_retries_it(self, metrics_workflow: dict[str, Any]) -> None:
        """Schedule must wait for completed Sunday data and retry the same week for three more days.

        PyPI Stats publishes no completion SLA, so a Monday run can fail on data that is not there yet. Every weekday in
        the Monday-to-Thursday window resolves to the same completed week, so the retries recover it; a Monday-only
        schedule would lose that week permanently, because the next Monday sees a non-contiguous history instead. The
        off-hour minute keeps the run away from the top-of-hour peak, when GitHub delays or drops scheduled runs.
        """
        triggers = metrics_workflow[True] if True in metrics_workflow else metrics_workflow["on"]

        assert triggers["schedule"] == [{"cron": "17 6 * * 1-4"}]
        assert "workflow_dispatch" in triggers

    def test_workflow_has_only_required_write_permissions(self, metrics_workflow: dict[str, Any]) -> None:
        """Automation must receive only permissions needed to update the metrics branch."""
        assert metrics_workflow["permissions"] == {"contents": "write"}

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

    def test_third_party_actions_are_pinned_to_commit_shas(self, metrics_workflow: dict[str, Any]) -> None:
        """Every third-party action must be pinned to an immutable commit SHA.

        This job holds write access to the repository, so a tag pin would let an upstream retag hand that access to code
        nobody here reviewed. Iterating every `uses:` also catches an action added later without a pin.
        """
        steps = metrics_workflow["jobs"]["update-metrics"]["steps"]

        unpinned = [step["uses"] for step in steps if "uses" in step and _sha_pinned_action(step["uses"]) is None]

        assert unpinned == []

    def test_steps_restore_then_generate_then_commit(self, metrics_workflow: dict[str, Any]) -> None:
        """Restore must precede generation, which must precede the commit.

        The generator reads its previous checkpoint out of the SVG on disk, so generating before restoring would re-
        baseline from the default branch, and committing before generating would publish the stale SVG.
        """
        names = [step["name"] for step in metrics_workflow["jobs"]["update-metrics"]["steps"]]

        assert names.index(RESTORE_STEP) < names.index(GENERATE_STEP) < names.index(COMMIT_STEP)

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

    def test_commit_step_updates_only_metrics_svg_on_the_automation_branch(
        self,
        metrics_steps: dict[str, dict[str, Any]],
    ) -> None:
        """Commit step must write only the generated SVG and force-update the fixed branch."""
        commit_step = metrics_steps[COMMIT_STEP]
        run = commit_step["run"]

        assert commit_step["env"]["METRICS_BRANCH"] == "automation/update-weekly-metrics"
        assert "git diff --quiet -- docs/assets/weekly-metrics.svg" in run
        assert "git add -- docs/assets/weekly-metrics.svg" in run
        assert 'git commit -m "docs: update weekly project metrics"' in run


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

    def test_lookup_failure_stops_instead_of_restarting_from_the_default_branch(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """A failed remote lookup must fail the step instead of masquerading as an absent branch.

        Treating an auth or network error as "no branch" would restart from the default branch's SVG and let the next
        push overwrite every unmerged checkpoint.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"],
            workspace,
            stubs,
            _restore_env(workspace, branch_exists=False, ls_remote_status=128),
        )

        assert result.returncode == 128
        assert "::error::git ls-remote failed for automation/update-weekly-metrics" in result.stdout
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG

    def test_fetch_failure_for_an_existing_branch_fails_the_step(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """A branch that exists but cannot be fetched must fail the step, not be treated as absent.

        The old `if git fetch` lumped every fetch failure in with "branch absent", so a transient error silently dropped
        the unmerged weeks from the next push.
        """
        workspace, stubs = restore_sandbox

        result = _run_step(
            metrics_steps[RESTORE_STEP]["run"],
            workspace,
            stubs,
            _restore_env(workspace, branch_exists=True, fetch_fails=True),
        )

        assert result.returncode == 1
        assert "::error::git fetch failed for automation/update-weekly-metrics" in result.stdout
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG

    def test_existing_branch_exports_its_tip_as_the_push_lease_base(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """The restored branch tip must reach the commit step through `GITHUB_ENV`.

        The push lease is anchored to the tip the SVG was restored from; a tip looked up at push time would match
        whatever the branch holds and degrade the lease into an unconditional force.
        """
        workspace, stubs = restore_sandbox

        _run_step(metrics_steps[RESTORE_STEP]["run"], workspace, stubs, _restore_env(workspace, branch_exists=True))

        assert (workspace / GITHUB_ENV_NAME).read_text(encoding="utf-8") == f"METRICS_BASE_SHA={STUB_BRANCH_SHA}\n"

    def test_restored_svg_is_staged_for_the_commit_guard(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        restore_sandbox: tuple[Path, Path],
    ) -> None:
        """The restored SVG must be staged so the no-op guard compares against the branch, not the default branch."""
        workspace, stubs = restore_sandbox

        _run_step(metrics_steps[RESTORE_STEP]["run"], workspace, stubs, _restore_env(workspace, branch_exists=True))

        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8")
        assert "git add -- docs/assets/weekly-metrics.svg" in logged


@requires_bash
class TestCommitAndPushStep:
    """Tests that run the commit step's own shell against a stubbed `git`."""

    def test_clean_svg_skips_commit_and_push(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
    ) -> None:
        """No SVG change must leave the automation branch untouched."""
        workspace, stubs = commit_sandbox

        result = _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=True, svg_changed=False),
        )

        assert result.returncode == 0, result.stderr
        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8")
        assert "commit -m" not in logged
        assert "push " not in logged

    def test_first_push_expects_the_branch_to_be_absent(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
    ) -> None:
        """A first run must create the automation branch behind an empty-expectation lease.

        An empty expected value makes git refuse the push when the branch has appeared since the restore step looked, so
        a concurrent creator is never overwritten.
        """
        workspace, stubs = commit_sandbox

        result = _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=False, svg_changed=True),
        )

        assert result.returncode == 0, result.stderr
        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8")
        assert (
            "push --force-with-lease=refs/heads/automation/update-weekly-metrics: "
            "origin HEAD:automation/update-weekly-metrics" in logged
        )

    def test_existing_branch_updates_with_force_with_lease(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
    ) -> None:
        """An existing automation branch must still be force-updated behind a lease."""
        workspace, stubs = commit_sandbox

        result = _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=True, svg_changed=True),
        )

        assert result.returncode == 0, result.stderr
        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8")
        assert (
            f"push --force-with-lease=refs/heads/automation/update-weekly-metrics:{STUB_BRANCH_SHA} "
            "origin HEAD:automation/update-weekly-metrics" in logged
        )

    @pytest.mark.parametrize("branch_exists", [True, False])
    def test_rejected_push_fails_the_step(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
        branch_exists: bool,
    ) -> None:
        """A push the remote refuses must turn the step red on both the lease and the first-push path.

        Losing the lease race (or hitting a protected branch) is exactly the signal that the branch did not receive the
        update; swallowing it, for example with `|| true`, would report a green run while the chart goes stale.
        """
        workspace, stubs = commit_sandbox

        result = _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=branch_exists, svg_changed=True, push_fails=True),
        )

        assert result.returncode != 0, result.stdout

    def test_commits_with_bot_identity_before_publishing(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
    ) -> None:
        """The step must set the bot identity, stage only the SVG, commit, and only then publish.

        A commit made before the identity is configured fails on a fresh runner, and a publish that precedes the commit
        would push whatever the checkout already held instead of the generated update.
        """
        workspace, stubs = commit_sandbox

        _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=True, svg_changed=True),
        )

        logged = (workspace / STUB_LOG_NAME).read_text(encoding="utf-8").splitlines()
        assert logged == [
            "git config user.name github-actions[bot]",
            "git config user.email 41898282+github-actions[bot]@users.noreply.github.com",
            "git diff --quiet -- docs/assets/weekly-metrics.svg",
            "git add -- docs/assets/weekly-metrics.svg",
            "git commit -m docs: update weekly project metrics",
            f"git push --force-with-lease=refs/heads/{METRICS_BRANCH}:{STUB_BRANCH_SHA} origin HEAD:{METRICS_BRANCH}",
        ]

    def test_pushed_update_prints_the_compare_link_for_a_maintainer(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        commit_sandbox: tuple[Path, Path],
    ) -> None:
        """A successful push must hand a maintainer the compare URL that opens the publishing pull request.

        Nothing in the workflow opens that pull request, so without the link the chart stays stale until someone notices
        the branch is ahead of the default branch.
        """
        workspace, stubs = commit_sandbox
        expected = "https://github.com/roboflow/rf-detr/compare/main...automation/update-weekly-metrics?expand=1"

        result = _run_step(
            metrics_steps[COMMIT_STEP]["run"],
            workspace,
            stubs,
            _commit_env(workspace, branch_exists=True, svg_changed=True),
        )

        assert result.returncode == 0, result.stderr
        assert f"::notice title=Weekly metrics pushed::Open a PR to publish: {expected}" in result.stdout
        assert expected in (workspace / STEP_SUMMARY_NAME).read_text(encoding="utf-8")


def _git(cwd: Path, *args: str) -> str:
    """Run an isolated `git` command and return its stripped stdout.

    Args:
        cwd: Directory the command runs in.
        *args: Arguments after `git`.

    Returns:
        Standard output of the command, stripped.

    Examples:
        >>> _git(Path("."), "--version").startswith("git version")
        True
    """
    return subprocess.run(
        ["git", *args], cwd=cwd, env=REAL_GIT_ENV, text=True, capture_output=True, check=True
    ).stdout.strip()


REAL_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


@pytest.fixture
def real_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    """Bare origin holding a default branch and an automation branch, plus a clone of the default branch.

    Returns the clone's workspace, the bare origin, and the automation branch's tip SHA.

    Examples:
        >>> real_remote  # doctest: +SKIP
        pytest fixture; builds a bare repository and a clone under tmp_path.
    """
    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))
    _git(seed, "init", "-b", "main")
    (seed / "docs" / "assets").mkdir(parents=True)
    (seed / TRACKED_SVG).write_text(CHECKED_IN_SVG, encoding="utf-8")
    _git(seed, "add", "--", TRACKED_SVG)
    _git(seed, "commit", "-m", "default branch svg")
    _git(seed, "checkout", "-b", METRICS_BRANCH)
    (seed / TRACKED_SVG).write_text(UNMERGED_SVG, encoding="utf-8")
    _git(seed, "commit", "-am", "unmerged svg")
    tip = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", origin.as_uri(), "main", METRICS_BRANCH)
    workspace = tmp_path / "workspace"
    _git(tmp_path, "clone", "--branch", "main", origin.as_uri(), str(workspace))
    return workspace, origin, tip


@requires_bash
@requires_git
class TestRestoreStepAgainstRealGit:
    """Tests that run the restore step's shell against a real repository and a bare origin."""

    @staticmethod
    def _restore(metrics_steps: dict[str, dict[str, Any]], workspace: Path) -> subprocess.CompletedProcess[str]:
        """Run the restore step with real git and a scratch `GITHUB_ENV` file."""
        env = {**REAL_GIT_ENV, "METRICS_BRANCH": METRICS_BRANCH, "GITHUB_ENV": str(workspace.parent / GITHUB_ENV_NAME)}
        return _run_step(metrics_steps[RESTORE_STEP]["run"], workspace, workspace, env)

    def test_existing_branch_is_restored_staged_and_exported(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        real_remote: tuple[Path, Path, str],
    ) -> None:
        """The branch's SVG must land staged in the workspace and its tip must be exported for the lease."""
        workspace, _, tip = real_remote

        result = self._restore(metrics_steps, workspace)

        assert result.returncode == 0, result.stderr
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == UNMERGED_SVG
        assert _git(workspace, "diff", "--cached", "--name-only") == TRACKED_SVG
        assert (workspace.parent / GITHUB_ENV_NAME).read_text(encoding="utf-8") == f"METRICS_BASE_SHA={tip}\n"

    def test_absent_branch_is_left_alone(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        real_remote: tuple[Path, Path, str],
    ) -> None:
        """With the automation branch deleted on the remote the step must change nothing and export nothing."""
        workspace, origin, _ = real_remote
        _git(origin, "branch", "-D", METRICS_BRANCH)

        result = self._restore(metrics_steps, workspace)

        assert result.returncode == 0, result.stderr
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG
        assert not (workspace.parent / GITHUB_ENV_NAME).exists()

    def test_unreachable_remote_fails_the_step(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        real_remote: tuple[Path, Path, str],
    ) -> None:
        """A remote that cannot be reached is a failure, not an absent branch."""
        workspace, origin, _ = real_remote
        shutil.rmtree(origin)

        result = self._restore(metrics_steps, workspace)

        assert result.returncode != 0
        assert "::error::git ls-remote failed" in result.stdout
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG

    def test_unfetchable_branch_fails_the_step(
        self,
        metrics_steps: dict[str, dict[str, Any]],
        real_remote: tuple[Path, Path, str],
    ) -> None:
        """A branch that `ls-remote` lists but `fetch` cannot serve must fail, not be treated as absent.

        Deleting the tip's loose object from the bare origin keeps the ref listed while the pack the fetch needs cannot
        be built, which is the shape of the transient fetch failure the old `if git fetch` swallowed.
        """
        workspace, origin, tip = real_remote
        (origin / "objects" / tip[:2] / tip[2:]).unlink()

        result = self._restore(metrics_steps, workspace)

        assert result.returncode != 0
        assert "::error::git fetch failed" in result.stdout
        assert (workspace / TRACKED_SVG).read_text(encoding="utf-8") == CHECKED_IN_SVG
