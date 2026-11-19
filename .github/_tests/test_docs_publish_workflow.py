# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for publishing and serving the root docs 404 page."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml


def test_publish_prepares_404_independently_of_the_docs_target(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prepare the root handler before publishing from a content tree without the asset."""
    workflow = yaml.safe_load((repo_root / ".github/workflows/publish-docs.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["doc-deploy"]["steps"]
    names = [step["name"] for step in steps]
    checkout_index = names.index("Check out root 404 from the workflow revision")
    prepare_index = names.index("Prepare the root 404 page")
    first_push = next(index for index, step in enumerate(steps) if "mike deploy --push" in step.get("run", ""))
    assert checkout_index < prepare_index < first_push

    checkout, prepare = steps[checkout_index], steps[prepare_index]
    assert checkout["with"]["ref"] == "${{ github.workflow_sha }}"
    assert checkout["if"] == prepare["if"] == "github.event_name != 'release' || !github.event.release.prerelease"
    asset = Path(checkout["with"]["sparse-checkout"].strip())
    assert asset.as_posix() == "docs/root-static/404.html"
    workflow_asset = tmp_path / checkout["with"]["path"] / asset
    workflow_asset.parent.mkdir(parents=True)
    workflow_asset.write_text("handler from workflow revision", encoding="utf-8")
    assert not (tmp_path / asset).exists()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))

    subprocess.run(
        ["bash", "-e", "-c", prepare["run"]],
        cwd=tmp_path,
        check=True,
    )

    deploy = steps[names.index("Deploy root static files to gh-pages")]["run"]
    assert 'cp "$RUNNER_TEMP/docs-root-404.html" 404.html' in deploy.splitlines()
    assert (runner_temp / "docs-root-404.html").read_text(encoding="utf-8") == "handler from workflow revision"


@pytest.mark.parametrize(
    ("path", "redirect"),
    [
        pytest.param("/%31.12.1/missing/", None, id="encoded-archive"),
        pytest.param("/%6Catest/missing/", None, id="encoded-latest"),
        pytest.param("/develop/missing/", None, id="literal-develop"),
        pytest.param("/1.12.1", None, id="version-without-trailing-slash"),
        pytest.param("/latest%2Fmissing/", "/latest/latest%2Fmissing/", id="encoded-separator"),
        pytest.param("/latest-guide/?q=1#anchor", "/latest/latest-guide/?q=1#anchor", id="lookalike"),
        pytest.param("/cookbooks/?q=1#anchor", "/latest/cookbooks/?q=1#anchor", id="unversioned"),
        pytest.param("/%zz/missing/?q=1#anchor", "/latest/%zz/missing/?q=1#anchor", id="malformed-escape"),
    ],
)
def test_root_404_preserves_versioned_paths(repo_root: Path, path: str, redirect: str | None) -> None:
    """Navigate only for unversioned paths and preserve their original URL components."""
    page = (repo_root / "docs/root-static/404.html").read_text(encoding="utf-8")
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        [
            "node",
            "-e",
            """
            const url = new URL(process.argv[1], "https://rfdetr.roboflow.com");
            const { pathname, search, hash } = url;
            const redirects = [];
            const window = { location: { pathname, search, hash, replace: value => redirects.push(value) } };
            require("node:vm").runInNewContext(require("node:fs").readFileSync(0, "utf8"), { window });
            process.stdout.write(JSON.stringify(redirects));
            """,
            path,
        ],
        input=script,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(result.stdout) == ([] if redirect is None else [redirect])
