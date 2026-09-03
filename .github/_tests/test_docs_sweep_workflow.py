# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the gh-pages sweep workflow that backfills archived doc versions.

The sweep is dispatched by hand against published documentation, so a mistake in it is only visible after it has
rewritten hundreds of live pages. These tests pin the wiring that a dry run cannot prove on its own.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def core_workflow(repo_root: Path) -> dict[str, Any]:
    """Parse the docs sweep workflow."""
    return yaml.safe_load((repo_root / ".github" / "workflows" / "docs-noindex-sweep.yml").read_text(encoding="utf-8"))


@pytest.fixture
def sweep_steps(core_workflow: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the sweep job steps in file order."""
    return core_workflow["jobs"]["sweep"]["steps"]


def step_named(steps: list[dict[str, Any]], fragment: str) -> dict[str, Any]:
    """Return the single step whose name contains a fragment.

    Args:
        steps: Steps of a workflow job.
        fragment: Substring identifying the wanted step.

    Returns:
        The matching step mapping.

    Examples:
        >>> step_named([{"name": "Inject noindex"}], "noindex")
        {'name': 'Inject noindex'}
    """
    matches = [step for step in steps if fragment in step.get("name", "")]
    assert len(matches) == 1, f"expected exactly one step matching {fragment!r}, found {len(matches)}"
    return matches[0]


def command_lines(run: str) -> str:
    """Return a run block's commands, with comment-only lines dropped.

    A shell comment may quote a command the step deliberately avoids, which would
    otherwise satisfy a plain substring search for that command.

    Args:
        run: Body of a workflow step's `run` block.

    Returns:
        The remaining lines, rejoined with newlines.

    Examples:
        >>> command_lines("git add -u\\n  # never git add -A")
        'git add -u'
    """
    return "\n".join(line for line in run.splitlines() if not line.strip().startswith("#"))


def test_injector_runs_after_the_noindex_rewrite(sweep_steps: list[dict[str, Any]]) -> None:
    names = [step.get("name", "") for step in sweep_steps]
    assert names.index("🏷️ Inject noindex") < names.index("🏷️ Inject outdated-version banner, styling, and offset script")


def test_injector_checkout_lands_in_a_subdirectory(sweep_steps: list[dict[str, Any]]) -> None:
    # Checking out over the gh-pages tree would discard everything the sweep patches.
    assert step_named(sweep_steps, "Checkout banner injector")["with"]["path"] == "_source"


def test_injector_checkout_uses_the_triggering_ref(sweep_steps: list[dict[str, Any]]) -> None:
    # A pinned ref would run develop's copy, so a pull_request dry run would test neither
    # the script the PR changes nor - before it merges - a script that exists at all.
    assert "ref" not in step_named(sweep_steps, "Checkout banner injector")["with"]


def test_injector_checkout_includes_the_script_it_copies(sweep_steps: list[dict[str, Any]]) -> None:
    sparse = step_named(sweep_steps, "Checkout banner injector")["with"]["sparse-checkout"]
    assert "docs/javascripts/version-banner.js" in sparse


def test_commit_stages_the_created_scripts(sweep_steps: list[dict[str, Any]]) -> None:
    # `git add -u` stages tracked files only, and the injector creates one JS file per
    # archived version; without this the pushed HTML would reference a missing script.
    assert "git add -- '[0-9]*/javascripts/version-banner.js'" in step_named(sweep_steps, "Commit and push")["run"]


def test_commit_never_stages_all_tracked_and_untracked(sweep_steps: list[dict[str, Any]]) -> None:
    # `git add -A` would also commit the _source checkout into gh-pages.
    assert "git add -A" not in command_lines(step_named(sweep_steps, "Commit and push")["run"])


def test_commit_never_stages_the_working_directory(sweep_steps: list[dict[str, Any]]) -> None:
    assert "git add ." not in command_lines(step_named(sweep_steps, "Commit and push")["run"])


def test_one_sweep_job_handles_both_triggers(core_workflow: dict[str, Any]) -> None:
    triggers = core_workflow[True] if True in core_workflow else core_workflow["on"]
    assert set(core_workflow["jobs"]) == {"sweep"}
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers


def test_push_is_gated_on_dispatch(core_workflow: dict[str, Any], sweep_steps: list[dict[str, Any]]) -> None:
    assert core_workflow["permissions"] == {"contents": "write"}
    assert step_named(sweep_steps, "Commit and push")["if"] == "github.event_name == 'workflow_dispatch'"


def test_dry_run_covers_the_injector(core_workflow: dict[str, Any]) -> None:
    triggers = core_workflow[True] if True in core_workflow else core_workflow["on"]
    assert ".github/scripts/inject_outdated_banner.py" in triggers["pull_request"]["paths"]
