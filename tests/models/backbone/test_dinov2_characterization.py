# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Characterization tests that pin the DINOv2 backbone's current behaviour.

The backbone classes inherit from ``transformers`` base classes that add hidden behaviour: weight initialization order,
the ``from_pretrained`` hub load on the meta device, gradient checkpointing set up from a config flag, tracing-aware
size casts and output validation. These tests record that behaviour on today's code, so replacing the base classes can
be checked against it without changing a test. Golden values come from ``_dinov2_goldens.py``, which documents how to
regenerate them.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.utils.checkpoint

from rfdetr.config import RFDETRNanoConfig, RFDETRSegNanoConfig
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.models.backbone.dinov2_with_windowed_attn import WindowedDinov2WithRegistersEmbeddings
from rfdetr.models.lwdetr import build_model_from_config
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
from tests._markers import _requires_huggingface_hub, _requires_peft, _requires_safetensors
from tests.models.backbone._dinov2_goldens import (
    ENCODER_CASES,
    FORWARD_GOLDEN,
    HUB_RNG_GOLDEN,
    INIT_GOLDEN,
    KEYS_GOLDEN,
    build_init_golden,
    encoder_key_shapes,
    forward_case_ids,
    lora_key_shapes,
    model_key_shapes,
    run_forward_case,
    tiny_backbone,
    tiny_config,
)

#: Exact float tolerance agreed for golden tensors: values may differ in the last bits across CPU architectures.
_GOLDEN_TOLERANCE = {"rtol": 0.0, "atol": 1e-6}

#: Hub revision of ``facebook/dinov2-small`` whose ``model.safetensors`` keys match the backbone exactly.
_DINOV2_SMALL_SHA = "ed25f3a31f01632728cabb09d1542f84ab7b0056"

#: ``DinoV2`` arguments of a patch-14 Nano-sized backbone, the geometry that loads DINOv2 hub weights.
_HUB_BACKBONE_KWARGS: dict[str, Any] = {
    "size": "small",
    "use_registers": False,
    "out_feature_indexes": [3, 6, 9, 12],
    "patch_size": 14,
    "positional_encoding_size": 37,
    "num_windows": 2,
}

#: Child process for the hub-path tests. ``huggingface_hub`` reads ``HF_HUB_CACHE`` and ``HF_HUB_OFFLINE`` when it is
#: imported, and the test session imports it during collection, so only a fresh process sees the fake cache.
_HUB_CHILD = textwrap.dedent(
    """
    import hashlib
    import json
    import sys

    import torch
    from safetensors.torch import load_file

    from rfdetr.config import RFDETRNanoConfig
    from rfdetr.models.backbone.dinov2 import DinoV2
    from rfdetr.models.lwdetr import build_model_from_config

    weights_path, kwargs = sys.argv[1], json.loads(sys.argv[2])
    torch.manual_seed(0)
    before = torch.get_rng_state()
    backbone = DinoV2(load_dinov2_weights=True, **kwargs)
    rng_untouched = torch.equal(before, torch.get_rng_state())
    expected = load_file(weights_path)
    loaded = backbone.encoder.state_dict()
    # ``load_file`` returns keys sorted by name, so compare the key sets, not their order.
    weights_equal = sorted(loaded) == sorted(expected) and all(torch.equal(loaded[k], expected[k]) for k in expected)
    torch.manual_seed(0)
    build_model_from_config(
        RFDETRNanoConfig(pretrain_weights=None, patch_size=14, positional_encoding_size=37, resolution=392)
    )
    report = {
        "rng_untouched": rng_untouched,
        "weights_equal": weights_equal,
        "training": backbone.encoder.training,
        "build_model_rng_digest": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }
    print(json.dumps(report))
    """
)

#: Child process for the no-``transformers`` test. A ``None`` entry in ``sys.modules`` makes every ``transformers``
#: import, submodules included, raise ``ImportError`` before ``rfdetr`` loads. The child then runs the core path:
#: build, predict, save and reload a checkpoint, and checks that no ``transformers`` module got loaded.
_NO_TRANSFORMERS_CHILD = textwrap.dedent(
    """
    import sys

    sys.modules["transformers"] = None

    import numpy as np
    import torch

    from rfdetr import RFDETRNano

    checkpoint_path = sys.argv[1]
    model = RFDETRNano(pretrain_weights=None, device="cpu")
    model.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    torch.save({"model": model.model.model.state_dict()}, checkpoint_path)
    RFDETRNano(pretrain_weights=checkpoint_path, device="cpu")
    leaked = sorted(
        name
        for name, module in sys.modules.items()
        if module is not None and (name == "transformers" or name.startswith("transformers."))
    )
    assert not leaked, leaked
    """
)


def _run_child(code: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python process and capture its output.

    Args:
        code: Python source passed to ``python -c``.
        *args: Command-line arguments, read by the child from ``sys.argv[1:]``.
        env: Child environment; ``None`` inherits the parent's.

    Returns:
        The finished process, whatever its return code.

    Examples:
        >>> _run_child("import sys; print(sys.argv[1])", "hello").stdout.strip()
        'hello'
    """
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )


def _parameter_grads(model: torch.nn.Module, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run a training-mode forward and backward pass and return every parameter gradient.

    Args:
        model: Backbone returning a tuple whose first item holds the feature maps.
        inputs: Image batch.

    Returns:
        Parameter name to gradient, for parameters that received one.

    Examples:
        >>> grads = _parameter_grads(tiny_backbone(), torch.ones(1, 3, 16, 16))
        >>> grads["embeddings.patch_embeddings.projection.weight"].shape
        torch.Size([8, 3, 4, 4])
    """
    model.train()
    loss = sum(feature_map.square().sum() for feature_map in model(inputs)[0])
    loss.backward()
    return {name: param.grad for name, param in model.named_parameters() if param.grad is not None}


@pytest.fixture(scope="module")
def keys_golden() -> dict[str, dict[str, str]]:
    """Key shapes per case from the committed key golden.

    Examples:
        A pytest fixture cannot be called directly.

        >>> keys_golden()  # doctest: +SKIP
    """
    return json.loads(KEYS_GOLDEN.read_text())["cases"]


@pytest.fixture(scope="module")
def init_golden() -> dict[str, Any]:
    """The committed random-init golden: tiny backbone tensors and the generator digest after the build.

    Examples:
        A pytest fixture cannot be called directly.

        >>> init_golden()  # doctest: +SKIP
    """
    return torch.load(INIT_GOLDEN, weights_only=True)


@pytest.fixture(scope="module")
def forward_golden() -> dict[str, Any]:
    """The committed forward golden: fixed inputs and the feature maps of every case.

    Examples:
        A pytest fixture cannot be called directly.

        >>> forward_golden()  # doctest: +SKIP
    """
    return torch.load(FORWARD_GOLDEN, weights_only=True)


class TestStateDictKeys:
    """State-dict keys and shapes match the golden: checkpoints, LoRA adapters and ONNX node names depend on them."""

    @pytest.mark.parametrize("encoder", list(ENCODER_CASES))
    def test_encoder_keys_match_golden(self, keys_golden: dict[str, dict[str, str]], encoder: str) -> None:
        """Each encoder name builds a backbone with the golden keys, in order, and shapes.

        A renamed module or a reordered registration changes the keys that released checkpoints and LoRA adapters are
        stored under.
        """
        assert list(encoder_key_shapes(encoder).items()) == list(keys_golden[encoder].items())

    @pytest.mark.parametrize(
        "case, model_config",
        [
            pytest.param("rfdetr_nano", RFDETRNanoConfig(pretrain_weights=None), id="rfdetr_nano"),
            pytest.param("rfdetr_seg_nano", RFDETRSegNanoConfig(pretrain_weights=None), id="rfdetr_seg_nano"),
        ],
    )
    def test_model_keys_match_golden(
        self, keys_golden: dict[str, dict[str, str]], case: str, model_config: Any
    ) -> None:
        """A full detection and segmentation model have the golden checkpoint keys, in order, and shapes."""
        assert list(model_key_shapes(model_config).items()) == list(keys_golden[case].items())

    @_requires_peft
    def test_lora_keys_match_golden(self, keys_golden: dict[str, dict[str, str]]) -> None:
        """A LoRA-wrapped Nano model has the golden keys, so ``backbone_lora=True`` checkpoints keep loading."""
        assert list(lora_key_shapes().items()) == list(keys_golden["rfdetr_nano_lora"].items())


class TestRandomInit:
    """Seeded random initialization of the backbone matches the golden, without hub weights."""

    def test_tensors_match_golden(self, init_golden: dict[str, Any]) -> None:
        """The tiny backbone built from seed 0 has the golden tensors.

        The initializer order and the random numbers each module draws decide every value, so a change in either shifts
        the tensors wholesale, far beyond the tolerance.
        """
        torch.testing.assert_close(build_init_golden()["state_dict"], init_golden["state_dict"], **_GOLDEN_TOLERANCE)

    def test_random_numbers_drawn_match_golden(self, init_golden: dict[str, Any]) -> None:
        """Building the tiny backbone leaves the generator in the golden state.

        Decoder and head initialization follow the backbone in the model builder, so extra or missing draws here change
        every later parameter for the same seed.
        """
        assert build_init_golden()["rng_digest"] == init_golden["rng_digest"]


class TestForward:
    """The backbone forward pass matches the golden feature maps."""

    @pytest.mark.parametrize("case_id", forward_case_ids())
    def test_feature_maps_match_golden(
        self, init_golden: dict[str, Any], forward_golden: dict[str, Any], case_id: str
    ) -> None:
        """Each attention implementation, window count and input size gives the golden feature maps.

        Weights come from the init golden, so this case does not depend on initialization. The resized input is non-
        square and runs position-embedding interpolation; the native input does not. The attention implementation is
        selected through ``config._attn_implementation``, which the config must keep settable.
        """
        actual = run_forward_case(case_id, init_golden["state_dict"], forward_golden["inputs"])

        torch.testing.assert_close(actual, forward_golden["outputs"][case_id], **_GOLDEN_TOLERANCE)


class TestGradientCheckpointing:
    """``gradient_checkpointing=True`` in the config sets up non-reentrant checkpointing on the encoder.

    No code calls ``gradient_checkpointing_enable``: the config flag is the only switch, consumed while the model is
    built. A spy on ``torch.utils.checkpoint.checkpoint`` cannot see the calls, because the function is bound when the
    model is built, so these tests check the bound function, the flags and the gradients.
    """

    @pytest.fixture
    def checkpointed(self) -> Any:
        """The tiny backbone built with gradient checkpointing.

        Examples:
            A pytest fixture cannot be called directly.

            >>> checkpointed()  # doctest: +SKIP
        """
        return tiny_backbone(gradient_checkpointing=True)

    def test_checkpoint_function_is_non_reentrant(self, checkpointed: Any) -> None:
        """The encoder calls ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False``."""
        function = checkpointed.encoder._gradient_checkpointing_func

        assert isinstance(function, functools.partial)
        assert function.func is torch.utils.checkpoint.checkpoint
        assert function.keywords == {"use_reentrant": False}

    def test_encoder_flag_is_set(self, checkpointed: Any) -> None:
        """The encoder's ``gradient_checkpointing`` attribute, which its forward pass reads, is true."""
        assert checkpointed.encoder.gradient_checkpointing is True

    def test_config_flag_is_consumed(self, checkpointed: Any) -> None:
        """The config no longer carries ``gradient_checkpointing`` once the model is built."""
        assert not hasattr(checkpointed.config, "gradient_checkpointing")

    def test_gradients_match_plain_run(self, checkpointed: Any) -> None:
        """Checkpointing recomputes activations but gives the same gradients as a run without it.

        The tiny config has no dropout or drop path, so the recomputation is exact.
        """
        plain = copy.deepcopy(checkpointed)
        plain.encoder.gradient_checkpointing = False
        inputs = torch.randn(1, 3, 16, 24)

        expected = _parameter_grads(plain, inputs)
        actual = _parameter_grads(checkpointed, inputs)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class TestTracedPositionEmbeddings:
    """Tracing the embeddings keeps the position-embedding interpolation size dynamic."""

    def test_traced_embeddings_follow_a_second_input_size(self) -> None:
        """Embeddings traced at one non-native size give eager results at another.

        The interpolation casts its sizes with a tracing-aware helper. Casting them with ``int()`` instead would bake
        the first size into the trace, and the second size would then produce a wrong-length output.
        """
        embeddings = WindowedDinov2WithRegistersEmbeddings(tiny_config()).eval()
        traced = torch.jit.trace(embeddings, torch.randn(1, 3, 24, 32), check_trace=False)
        inputs = torch.randn(1, 3, 32, 20)

        torch.testing.assert_close(traced(inputs), embeddings(inputs), **_GOLDEN_TOLERANCE)


class TestOutFeatureIndexesValidation:
    """Output-stage requests that disagree with the encoder's JSON config raise instead of building silently."""

    def test_registers_encoder_with_multi_stage_request_raises(self) -> None:
        """A registers encoder asked for four stages raises, because its JSON config pins a single-stage index list.

        This records a known limitation: the JSON's ``out_indices`` stays stale when only ``out_features`` is
        overridden. Fixing it is a separate change; until then the mismatch must raise, not build a wrong backbone.
        """
        with pytest.raises(ValueError, match="same length"):
            DinoV2(size="small", use_registers=True, out_feature_indexes=[2, 5, 8, 11], load_dinov2_weights=False)


class TestLoraForward:
    """LoRA adapters wrap the backbone encoder and the model still runs."""

    @_requires_peft
    def test_lora_wrapped_model_runs_forward(self) -> None:
        """A Nano model with LoRA adapters on its backbone runs a training-mode forward pass to finite class logits."""
        from rfdetr.models.weights import apply_lora

        model = build_model_from_config(RFDETRNanoConfig(pretrain_weights=None))
        apply_lora(model)
        samples = nested_tensor_from_tensor_list([torch.randn(3, 384, 384)])

        outputs = model(samples)

        assert outputs["pred_logits"].isfinite().all()


@pytest.fixture(scope="module")
def hub_child_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build the backbone from a fake offline Hugging Face cache in a child process and return its report.

    The cache holds ``facebook/dinov2-small`` at its pinned revision, with random tensors at the real shapes, so the
    load runs end to end without the network. Both the ``refs/main`` pointer and the pinned snapshot are written.

    Examples:
        A pytest fixture cannot be called directly.

        >>> hub_child_report(tmp_path_factory)  # doctest: +SKIP
    """
    from safetensors.torch import save_file

    cache = tmp_path_factory.mktemp("hf_cache")
    repo = cache / "models--facebook--dinov2-small"
    snapshot = repo / "snapshots" / _DINOV2_SMALL_SHA
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(_DINOV2_SMALL_SHA, newline="\n")
    with torch.device("meta"):
        shapes = DinoV2(load_dinov2_weights=False, **_HUB_BACKBONE_KWARGS).encoder.state_dict()
    generator = torch.Generator().manual_seed(1)
    weights = {key: torch.randn(value.shape, generator=generator) for key, value in shapes.items()}
    weights_path = snapshot / "model.safetensors"
    save_file(weights, str(weights_path), metadata={"format": "pt"})
    env = {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"}

    result = _run_child(_HUB_CHILD, str(weights_path), json.dumps(_HUB_BACKBONE_KWARGS), env=env)

    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@_requires_huggingface_hub
@_requires_safetensors
class TestHubWeightsLoad:
    """Loading DINOv2 hub weights builds on the meta device, loads the file as is and returns an eval-mode encoder."""

    def test_load_draws_no_random_numbers(self, hub_child_report: dict[str, Any]) -> None:
        """Building the backbone with hub weights leaves the generator untouched.

        The weights replace every parameter, so drawing random initial values would only shift the decoder and head
        initialization that follows in the model builder.
        """
        assert hub_child_report["rng_untouched"] is True

    def test_loaded_weights_equal_the_file(self, hub_child_report: dict[str, Any]) -> None:
        """The encoder state dict equals the hub file: same keys, same order, same values."""
        assert hub_child_report["weights_equal"] is True

    def test_loaded_encoder_is_in_eval_mode(self, hub_child_report: dict[str, Any]) -> None:
        """The loaded encoder starts in eval mode, as ``from_pretrained`` returns it."""
        assert hub_child_report["training"] is False

    def test_model_build_draws_golden_random_numbers(self, hub_child_report: dict[str, Any]) -> None:
        """Building a patch-14 Nano model that loads hub weights leaves the generator in the golden state.

        Together with the backbone drawing nothing, this pins decoder and head initialization for a fixed seed without
        storing their tensors.
        """
        golden = json.loads(HUB_RNG_GOLDEN.read_text())

        assert hub_child_report["build_model_rng_digest"] == golden["rng_digest"]


class TestCorePathWithoutTransformers:
    """The core path (import, build, predict, checkpoint round trip) runs with ``transformers`` unimportable."""

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="transformers is still imported on the core path; the backbone rewrite removes it",
    )
    def test_core_path_runs_with_transformers_blocked(self, tmp_path: Path) -> None:
        """A child process that cannot import ``transformers`` builds Nano, predicts and reloads a saved checkpoint.

        The checkpoint is written to a temporary directory and loaded from there, so nothing is downloaded. Today the
        child fails at ``import rfdetr``; the parent then fails the return-code assertion, which the strict expected
        failure records.
        """
        result = _run_child(_NO_TRANSFORMERS_CHILD, str(tmp_path / "checkpoint.pth"))

        assert result.returncode == 0, result.stderr[-4000:]
