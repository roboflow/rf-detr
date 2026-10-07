# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the pretrained-weights Actions cache shared across CI workflows.

One cache entry is written by a single `try-all-models` leg on push to develop and restored by jobs in three workflows.
Nothing fails when this wiring drifts: a PR run that starts writing entries evicts other caches, a writer leg that
restores never exercises the live download and MD5 check again, and a reader whose key no longer matches quietly falls
back to downloading. These tests pin the contract that CI results cannot show.
"""

import itertools
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WEIGHTS_PATH = ".rfdetr-weights"
RF_HOME = "${{ github.workspace }}/.rfdetr-weights"
CACHE_ACTION = re.compile(r"actions/cache(?:/(?P<mode>restore|save))?@(?P<ref>\S+)")
WRITER = ("ci-integrations.yml", "try-all-models")
FULL_KEY_READER = ("ci-tests-gpu.yml", "run-gpu-tests")
PREFIX_READERS = [("ci-integrations.yml", "export-parity"), ("ci-legacy-checkpoints.yml", "generate")]
CACHE_JOBS = [WRITER, FULL_KEY_READER, *PREFIX_READERS]
PREFIX_READER_PARAMS = [pytest.param(job, id=f"{job[0]}:{job[1]}") for job in PREFIX_READERS]
CACHE_JOB_PARAMS = [pytest.param(job, id=f"{job[0]}:{job[1]}") for job in CACHE_JOBS]
FULL_KEY = "${{ steps.weights-key.outputs.key }}"
REGISTRY_PREFIX = "rfdetr-weights-${{ steps.weights-key.outputs.registry }}-"


def command_lines(run: str) -> str:
    """Return a run block's commands, with comment-only lines dropped.

    Args:
        run: Body of a workflow step's `run` block.

    Returns:
        The remaining lines, stripped and rejoined with newlines.

    Examples:
        >>> command_lines("  # explain\\n  echo hi\\n")
        'echo hi'
    """
    return "\n".join(line.strip() for line in run.splitlines() if line.strip() and not line.strip().startswith("#"))


def step_named(steps: list[dict[str, Any]], fragment: str) -> dict[str, Any]:
    """Return the single step whose name contains a fragment.

    Args:
        steps: Steps of a workflow job.
        fragment: Substring identifying the wanted step.

    Returns:
        The matching step mapping.

    Examples:
        >>> step_named([{"name": "📦 Restore cached model weights"}], "Restore cached model weights")
        {'name': '📦 Restore cached model weights'}
    """
    matches = [step for step in steps if fragment in step.get("name", "")]
    assert len(matches) == 1, f"expected exactly one step matching {fragment!r}, found {len(matches)}"
    return matches[0]


def weights_cache_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the `actions/cache` steps that operate on the weights directory.

    Args:
        steps: Steps of a workflow job.

    Returns:
        The cache steps whose `path` is the weights directory, in file order.

    Examples:
        >>> weights_cache_steps([
        ...     {"uses": "actions/cache/restore@abc", "with": {"path": ".rfdetr-weights"}},
        ...     {"uses": "actions/cache@abc", "with": {"path": "~/.cache/huggingface"}},
        ...     {"run": "echo"},
        ... ])
        [{'uses': 'actions/cache/restore@abc', 'with': {'path': '.rfdetr-weights'}}]
    """
    return [
        step
        for step in steps
        if CACHE_ACTION.match(step.get("uses", "")) and step.get("with", {}).get("path") == WEIGHTS_PATH
    ]


def gate_clauses(expression: str) -> dict[str, str]:
    """Split a `${{ a == 'x' && b == 'y' }}` gate into its required context values.

    Only a conjunction of equality checks is accepted, so a later edit that adds an `||` or a `!=` (which could
    widen who writes the cache) fails here instead of being silently misread.

    Args:
        expression: A GitHub Actions expression made of `context == 'literal'` clauses joined by `&&`.

    Returns:
        Mapping of each context path to the literal it must equal.

    Examples:
        >>> gate_clauses("${{ github.event_name == 'push' && matrix.os == 'ubuntu-latest' }}")
        {'github.event_name': 'push', 'matrix.os': 'ubuntu-latest'}
    """
    body = expression.strip()
    assert body.startswith("${{") and body.endswith("}}"), f"not a single expression: {expression!r}"
    clauses = {}
    for clause in body[3:-2].split("&&"):
        match = re.fullmatch(r"\s*([\w.-]+)\s*==\s*'([^']*)'\s*", clause)
        assert match, f"gate clause is not a plain equality check: {clause!r}"
        clauses[match.group(1)] = match.group(2)
    return clauses


def gate_passes(clauses: dict[str, str], context: dict[str, str]) -> bool:
    """Evaluate a gate parsed by :func:`gate_clauses` against concrete run values.

    Args:
        clauses: Required context values of the gate.
        context: Values of the same context paths for one run.

    Returns:
        Whether every clause holds for that run.

    Examples:
        >>> gate_passes({"github.event_name": "push"}, {"github.event_name": "pull_request"})
        False
    """
    return all(context[path] == value for path, value in clauses.items())


@pytest.fixture(scope="module")
def workflows(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Parse every workflow in the repository, keyed by file name."""
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((repo_root / ".github" / "workflows").glob("*.yml"))
    }


@pytest.fixture(scope="module")
def writer_job(workflows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The `try-all-models` job, the only one allowed to write the weights cache."""
    return workflows[WRITER[0]]["jobs"][WRITER[1]]


@pytest.fixture(scope="module")
def writer_gate(writer_job: dict[str, Any]) -> dict[str, str]:
    """Context values that make a `try-all-models` leg the cache writer."""
    return gate_clauses(writer_job["env"]["WEIGHTS_CACHE_WRITER"])


class TestWriterGate:
    """Exactly one leg, and only on push to develop, may write the cache."""

    WRITER_RUN = {
        "github.event_name": "push",
        "github.ref": "refs/heads/develop",
        "matrix.os": "ubuntu-latest",
        "matrix.python-version": "3.13",
    }

    def test_push_to_develop_on_the_writer_leg_writes(self, writer_gate: dict[str, str]) -> None:
        assert gate_passes(writer_gate, self.WRITER_RUN)

    @pytest.mark.parametrize(
        "override",
        [
            pytest.param({"github.event_name": "pull_request", "github.ref": "refs/pull/1/merge"}, id="pull_request"),
            pytest.param({"github.ref": "refs/heads/main"}, id="push_to_main"),
            pytest.param({"github.ref": "refs/heads/release/latest"}, id="push_to_release"),
            pytest.param({"matrix.os": "windows-latest"}, id="other_os"),
            pytest.param({"matrix.python-version": "3.10"}, id="other_python"),
        ],
    )
    def test_any_other_run_does_not_write(self, writer_gate: dict[str, str], override: dict[str, str]) -> None:
        assert not gate_passes(writer_gate, {**self.WRITER_RUN, **override})

    def test_writer_leg_is_in_the_matrix(self, writer_job: dict[str, Any], writer_gate: dict[str, str]) -> None:
        matrix = writer_job["strategy"]["matrix"]
        legs = set(itertools.product(matrix["os"], matrix["python-version"]))
        assert (writer_gate["matrix.os"], writer_gate["matrix.python-version"]) in legs


class TestWriterSteps:
    """The writer downloads cold, then saves only a new key."""

    def test_writer_leg_skips_the_restore(self, writer_job: dict[str, Any]) -> None:
        # Restoring here would stop the live download and MD5 check from running anywhere.
        assert (
            step_named(writer_job["steps"], "Restore cached model weights")["if"]
            == "env.WEIGHTS_CACHE_WRITER != 'true'"
        )

    def test_lookup_runs_only_on_the_writer_leg(self, writer_job: dict[str, Any]) -> None:
        lookup = step_named(writer_job["steps"], "already cached")
        assert lookup["if"] == "env.WEIGHTS_CACHE_WRITER == 'true'"

    def test_lookup_downloads_nothing(self, writer_job: dict[str, Any]) -> None:
        assert step_named(writer_job["steps"], "already cached")["with"]["lookup-only"] is True

    def test_save_requires_the_writer_leg_and_a_new_key(self, writer_job: dict[str, Any]) -> None:
        lookup_id = step_named(writer_job["steps"], "already cached")["id"]
        save = step_named(writer_job["steps"], "Save model weights cache")
        assert save["if"] == f"env.WEIGHTS_CACHE_WRITER == 'true' && steps.{lookup_id}.outputs.cache-hit != 'true'"

    def test_save_runs_after_the_smoke_test(self, writer_job: dict[str, Any]) -> None:
        # A save before the smoke test could cache files that never passed the MD5 check.
        names = [step.get("name", "") for step in writer_job["steps"]]
        smoke = names.index(step_named(writer_job["steps"], "Smoke-test")["name"])
        assert smoke < names.index(step_named(writer_job["steps"], "Save model weights cache")["name"])

    @pytest.mark.parametrize("fragment", ["Restore cached model weights", "already cached", "Save model weights cache"])
    def test_writer_steps_use_the_full_key(self, writer_job: dict[str, Any], fragment: str) -> None:
        assert step_named(writer_job["steps"], fragment)["with"]["key"] == FULL_KEY

    def test_writer_restore_has_no_fallback(self, writer_job: dict[str, Any]) -> None:
        # A prefix fallback would restore weights cached for another rfdetr_plus version.
        assert "restore-keys" not in step_named(writer_job["steps"], "Restore cached model weights")["with"]


class TestSaveIsConfinedToTheWriter:
    """No other job, in any workflow, writes the weights cache."""

    def test_only_one_step_saves_weights(self, workflows: dict[str, dict[str, Any]]) -> None:
        savers = [
            (name, job_id)
            for name, workflow in workflows.items()
            for job_id, job in workflow.get("jobs", {}).items()
            for step in weights_cache_steps(job.get("steps", []))
            if CACHE_ACTION.match(step["uses"]).group("mode") != "restore"
        ]
        # `actions/cache` without /restore saves in its post step, so it counts as a writer too.
        assert savers == [WRITER]


class TestSharedKey:
    """Every reader resolves the key the writer saved."""

    def test_full_key_reader_computes_the_writer_key(self, workflows: dict[str, dict[str, Any]]) -> None:
        writer = step_named(workflows[WRITER[0]]["jobs"][WRITER[1]]["steps"], "Compute model weights cache key")
        reader = step_named(
            workflows[FULL_KEY_READER[0]]["jobs"][FULL_KEY_READER[1]]["steps"], "Compute model weights cache key"
        )
        assert command_lines(reader["run"]) == command_lines(writer["run"])

    def test_full_key_reader_restores_the_exact_key(self, workflows: dict[str, dict[str, Any]]) -> None:
        steps = workflows[FULL_KEY_READER[0]]["jobs"][FULL_KEY_READER[1]]["steps"]
        assert step_named(steps, "Restore cached model weights")["with"]["key"] == FULL_KEY

    def test_full_key_reader_has_no_fallback(self, workflows: dict[str, dict[str, Any]]) -> None:
        steps = workflows[FULL_KEY_READER[0]]["jobs"][FULL_KEY_READER[1]]["steps"]
        assert "restore-keys" not in step_named(steps, "Restore cached model weights")["with"]

    def test_writer_key_starts_with_the_registry_prefix(self, writer_job: dict[str, Any]) -> None:
        # Prefix readers only find the entry if the registry hash comes first in the saved key.
        run = command_lines(step_named(writer_job["steps"], "Compute model weights cache key")["run"])
        assert 'key=rfdetr-weights-${registry}-plus-${plus}"' in run

    @pytest.mark.parametrize("workflow_job", PREFIX_READER_PARAMS)
    def test_prefix_reader_hashes_the_same_registry(
        self, workflows: dict[str, dict[str, Any]], writer_job: dict[str, Any], workflow_job: tuple[str, str]
    ) -> None:
        writer_run = command_lines(step_named(writer_job["steps"], "Compute model weights cache key")["run"])
        registry = re.search(r"registry=\$\((git rev-parse [^)]+)\)", writer_run).group(1)
        steps = workflows[workflow_job[0]]["jobs"][workflow_job[1]]["steps"]
        reader_run = command_lines(step_named(steps, "Compute model weights cache key")["run"])
        assert f"registry=$({registry})" in reader_run

    @pytest.mark.parametrize("workflow_job", PREFIX_READER_PARAMS)
    def test_prefix_reader_falls_back_to_the_registry_prefix(
        self, workflows: dict[str, dict[str, Any]], workflow_job: tuple[str, str]
    ) -> None:
        steps = workflows[workflow_job[0]]["jobs"][workflow_job[1]]["steps"]
        restore_keys = step_named(steps, "Restore cached model weights")["with"]["restore-keys"]
        assert restore_keys.strip().splitlines() == [REGISTRY_PREFIX]


@pytest.mark.parametrize("workflow_job", CACHE_JOB_PARAMS)
class TestEveryCacheJob:
    """Settings that are part of the cache version, so every job must agree on them."""

    def test_job_points_rf_home_at_the_cached_directory(
        self, workflows: dict[str, dict[str, Any]], workflow_job: tuple[str, str]
    ) -> None:
        assert workflows[workflow_job[0]]["jobs"][workflow_job[1]]["env"]["RF_HOME"] == RF_HOME

    def test_job_has_weights_cache_steps(
        self, workflows: dict[str, dict[str, Any]], workflow_job: tuple[str, str]
    ) -> None:
        assert weights_cache_steps(workflows[workflow_job[0]]["jobs"][workflow_job[1]]["steps"])

    def test_cache_steps_share_one_entry_across_operating_systems(
        self, workflows: dict[str, dict[str, Any]], workflow_job: tuple[str, str]
    ) -> None:
        steps = weights_cache_steps(workflows[workflow_job[0]]["jobs"][workflow_job[1]]["steps"])
        assert all(step["with"].get("enableCrossOsArchive") is True for step in steps)

    def test_cache_steps_pin_the_writer_action_version(
        self, workflows: dict[str, dict[str, Any]], writer_job: dict[str, Any], workflow_job: tuple[str, str]
    ) -> None:
        writer_refs = {
            CACHE_ACTION.match(step["uses"]).group("ref") for step in weights_cache_steps(writer_job["steps"])
        }
        steps = weights_cache_steps(workflows[workflow_job[0]]["jobs"][workflow_job[1]]["steps"])
        assert {CACHE_ACTION.match(step["uses"]).group("ref") for step in steps} == writer_refs


def test_legacy_links_cached_weights_before_generating(workflows: dict[str, dict[str, Any]]) -> None:
    # rfdetr < 1.7 looks for weights in the working directory and ignores RF_HOME.
    names = [step.get("name", "") for step in workflows["ci-legacy-checkpoints.yml"]["jobs"]["generate"]["steps"]]
    restore = names.index("📦 Restore cached model weights")
    link = names.index("🔗 Expose cached weights to releases that ignore RF_HOME")
    generate = names.index("🏗️ Generate checkpoint")
    assert restore < link < generate
