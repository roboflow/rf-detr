# Legacy Checkpoint Backward-Compatibility Tests

Verifies that the *current* codebase can load RF-DETR checkpoints produced by every past stable minor release listed in `checkpoint_versions.txt`. Runs as a dedicated GitHub Actions workflow (`.github/workflows/ci-legacy-checkpoints.yml`), separate from the standard CPU/GPU test suites (both exclude `tests/legacy/` via `--ignore`).

## How it works

1. `get-versions` job reads `checkpoint_versions.txt` and exposes it as a matrix.
2. `generate` job (one per version) installs that historical `rfdetr` release from its frozen dependency set, runs `generate_checkpoint.py` to produce a `.pth` checkpoint in that version's save format, and uploads it as a workflow artifact.
3. `test-compat` job downloads every generated checkpoint, installs the *current* (dev) code, and runs `test_checkpoint_compat.py` against each one — asserting a successful load, correct model structure, and (for `--use-pretrained` checkpoints) that a reference prediction is reproduced after reload.

## Adding a new legacy version

1. Add the version to `checkpoint_versions.txt` (one per line; comment lines and blank lines are ignored).
2. Add a matching `frozen_dependencies/req-<version>.txt` pinning `rfdetr==<version>` plus any transitive dependency pin needed to keep the install reproducible (see an existing `req-*.txt` for the header/format convention). `test_every_version_has_matching_frozen_requirements_file` in `test_checkpoint_compat.py` fails locally if this file is missing, instead of only failing much later at CI matrix install time.
3. Run `uv run --no-sync pytest tests/legacy/ -v` locally once checkpoints are available (or rely on the CI workflow, which generates them).

## Why `frozen_dependencies/` is a *partial* freeze, not a full lockfile

Each `req-<version>.txt` pins `rfdetr==<version>` and a bounded `transformers` range (the narrowest range still compatible with that historical release, documented per-file), but leaves `numpy`/`torch`/`supervision`/`pillow` unpinned. This is deliberate: those are shared, frequently-updated dependencies where full pinning would require maintaining N separate lock files indefinitely. The tradeoff is that a future major bump in one of the unpinned packages could occasionally break an older `generate` job for reasons unrelated to the checkpoint-format compatibility this suite exists to test — if that happens, add a targeted pin to the affected version's `req-*.txt` (see `req-1.4.3.txt` for an example of a targeted `transformers` bound added for exactly this reason).

## Pretrained-checkpoint fetch: skip vs. fail

`generate_checkpoint.py --use-pretrained` downloads real COCO-pretrained weights. Network/5xx/ timeout/DNS errors during that fetch are treated as infra outages (skip), while a genuine permanent absence (e.g. a 404 for a removed asset) still fails the test — a blanket skip would silently green a real regression. This mirrors the reference-image fetch's existing MD5-guard skip-vs-fail posture in `generate_checkpoint.py`.
