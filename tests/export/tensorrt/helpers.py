# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared builders for the TensorRT export tests: engine-builder fakes and synthetic INT8 ONNX graphs.

The INT8 builders import ``onnx`` inside each function, so importing this module never needs it.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rfdetr.export._tensorrt import quantize
from rfdetr.export.prepare import ExportGraph

__doctest_requires__ = {
    ("_outputs_before_and_after_lift",): ["onnx", "onnxruntime"],
    (
        "_attention_and_mlp",
        "_weight",
    ): ["onnx"],
}


def _fake_tensorrt(version: str, *, has_fp16_flag: bool) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module reporting *version* and optionally lacking ``BuilderFlag.FP16``.

    Lets the version-dependent branches in ``build_engine`` be exercised on a machine with no TensorRT
    at all, including the TensorRT 11 shape where the flag was removed from the API.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        has_fp16_flag: Whether ``BuilderFlag`` should carry an ``FP16`` member.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> module = _fake_tensorrt("11.2.1.2", has_fp16_flag=False)
        >>> module.__version__
        '11.2.1.2'
        >>> hasattr(module.BuilderFlag, "FP16")
        False
        >>> hasattr(_fake_tensorrt("10.16.1.11", has_fp16_flag=True).BuilderFlag, "FP16")
        True
    """
    module = types.ModuleType("tensorrt")
    module.__version__ = version

    class BuilderFlag:
        INT8 = 0

    if has_fp16_flag:
        BuilderFlag.FP16 = 1
    module.BuilderFlag = BuilderFlag
    return module


def _minimal_export_graph() -> ExportGraph:
    """Build the smallest static `ExportGraph` `TensorRTExporter._convert()` needs, without a real RF-DETR model.

    Examples:
        >>> graph = _minimal_export_graph()
        >>> graph.backbone_only
        False
        >>> graph.input_tensors.shape
        torch.Size([1, 3, 8, 8])
    """
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 3, 8, 8),
        input_names=("input",),
        output_names=("dets",),
        dynamic_axes=None,
        shape=(8, 8),
        backbone_only=False,
    )


@dataclass(frozen=True)
class _FakeNetworkInput:
    """One parsed network input: a name and a TensorRT-style shape (``-1`` marks the dynamic batch axis)."""

    name: str
    shape: tuple[int, ...]


class _FakeNetwork:
    """Stand-in for a parsed ``trt.INetworkDefinition`` exposing only ``num_inputs`` / ``get_input``."""

    def __init__(self, *inputs: _FakeNetworkInput) -> None:
        self._inputs = inputs

    @property
    def num_inputs(self) -> int:
        return len(self._inputs)

    def get_input(self, index: int) -> _FakeNetworkInput:
        return self._inputs[index]


#: The one input an RF-DETR graph exported without ``dynamic_batch`` has: every axis fixed at trace time.
_STATIC_INPUT = _FakeNetworkInput("input", (1, 3, 384, 384))


class _FakeProfile:
    """Stand-in for ``polygraphy.backend.trt.Profile`` recording every ``add`` call."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, tuple[int, ...]]] = {}

    def add(self, name: str, min: tuple[int, ...], opt: tuple[int, ...], max: tuple[int, ...]) -> _FakeProfile:
        self.entries[name] = {"min": min, "opt": opt, "max": max}
        return self


#: Node-name prefix of a block the plan treats as backbone encoder (INT8 attention allowed).
BACKBONE = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/"

#: Node-name prefix of a block the plan treats as decoder (INT8 linears, FP16 attention).
DECODER = "/transformer/decoder/layers.0/"

#: Node-name prefix outside every quantized region (a detection head).
HEAD = "/class_embed/"


def _weight(name: str, rows: int, cols: int, dtype: type = np.float32, seed: int = 0) -> Any:
    """Return a seeded random ``rows x cols`` weight initializer.

    Examples:
        >>> _weight("w", 2, 3).dims
        [2, 3]
    """
    from onnx import numpy_helper

    values = np.random.default_rng(seed).standard_normal((rows, cols)) * 0.1
    return numpy_helper.from_array(values.astype(dtype), name)


def _attention_and_mlp(
    attention_prefix: str,
    *,
    sequence: int,
    head_size: int,
    mlp_prefix: str = BACKBONE,
    second_mlp_prefix: str | None = None,
    activation: str = "Relu",
    dtype: type = np.float32,
    scaled_scores: bool = False,
    second_reader: bool = False,
) -> Any:
    """Build ``x -> attention block -> MLP`` with RF-DETR-style node names, as a ``ModelProto``.

    The attention block projects ``x`` to q/k/v, splits two heads, runs ``MatMul -> Softmax -> MatMul``, projects back;
    the MLP is ``fc1 -> activation -> fc2``. ``activation="Erf"`` writes GELU in its traced erf form, ``"Gelu"`` as the
    opset-20 op.

    Args:
        attention_prefix: Node-name prefix of the attention block.
        sequence: Tokens per sequence.
        head_size: Channels per head (two heads).
        mlp_prefix: Node-name prefix of ``fc1``.
        second_mlp_prefix: Node-name prefix of ``fc2``; defaults to *mlp_prefix*.
        activation: ``"Relu"``, ``"Erf"`` or ``"Gelu"``.
        dtype: Float type of the graph (``np.float32`` or ``np.float16``).
        scaled_scores: Divide the attention scores by a constant before the ``Softmax`` (eager attention's form).
        second_reader: Give the merged attention output a second reader besides the output projection.

    Returns:
        A checked ``ModelProto``.

    Examples:
        >>> model = _attention_and_mlp(BACKBONE, sequence=4, head_size=16)
        >>> sum(node.op_type == "Softmax" for node in model.graph.node)
        1
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    width = 2 * head_size
    elem = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    second_mlp_prefix = second_mlp_prefix or mlp_prefix
    inits = [_weight(f"w{n}", width, width, dtype, seed) for seed, n in enumerate("qkvo")]
    inits += [_weight("w1", width, 2 * width, dtype, 5), _weight("w2", 2 * width, width, dtype, 6)]
    inits += [
        numpy_helper.from_array(np.array(v, dtype), n) for n, v in (("sqrt2", 1.4142), ("one", 1.0), ("half", 0.5))
    ]
    inits += [
        numpy_helper.from_array(np.array([1, sequence, 2, head_size], np.int64), "heads"),
        numpy_helper.from_array(np.array([1, sequence, width], np.int64), "merged"),
    ]
    p = attention_prefix
    nodes = []
    for name in "qkv":
        nodes += [
            helper.make_node("MatMul", ["x", f"w{name}"], [f"{name}0"], f"{p}{name}/MatMul"),
            helper.make_node("Reshape", [f"{name}0", "heads"], [f"{name}1"], f"{p}{name}/Reshape"),
        ]
    nodes += [
        helper.make_node("Transpose", ["q1"], ["q2"], f"{p}q/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k1"], ["k2"], f"{p}k/Transpose", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v1"], ["v2"], f"{p}v/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q2", "k2"], ["s0" if scaled_scores else "s"], f"{p}scores/MatMul"),
        *([helper.make_node("Div", ["s0", "sqrt2"], ["s"], f"{p}scale/Div")] if scaled_scores else []),
        helper.make_node("Softmax", ["s"], ["p"], f"{p}Softmax", axis=-1),
        helper.make_node("MatMul", ["p", "v2"], ["c"], f"{p}context/MatMul"),
        helper.make_node("Transpose", ["c"], ["c1"], f"{p}merge/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["c1", "merged"], ["c2"], f"{p}merge/Reshape"),
        helper.make_node("MatMul", ["c2", "wo"], ["a"], f"{p}o/MatMul"),
        helper.make_node("MatMul", ["a", "w1"], ["h"], f"{mlp_prefix}fc1/MatMul"),
    ]
    if activation == "Erf":
        nodes += [
            helper.make_node("Div", ["h", "sqrt2"], ["h1"], f"{mlp_prefix}act/Div"),
            helper.make_node("Erf", ["h1"], ["h2"], f"{mlp_prefix}act/Erf"),
            helper.make_node("Add", ["h2", "one"], ["h3"], f"{mlp_prefix}act/Add"),
            helper.make_node("Mul", ["h", "h3"], ["h4"], f"{mlp_prefix}act/Mul"),
            helper.make_node("Mul", ["h4", "half"], ["g"], f"{mlp_prefix}act/Mul_1"),
        ]
    else:
        nodes.append(helper.make_node(activation, ["h"], ["g"], f"{mlp_prefix}act/{activation}"))
    nodes.append(helper.make_node("MatMul", ["g", "w2"], ["y"], f"{second_mlp_prefix}fc2/MatMul"))
    outputs = [helper.make_tensor_value_info("y", elem, [1, sequence, width])]
    if second_reader:
        nodes.append(helper.make_node("Identity", ["c2"], ["attention_out"], f"{p}tap/Identity"))
        outputs.append(helper.make_tensor_value_info("attention_out", elem, [1, sequence, width]))
    graph = helper.make_graph(
        nodes, "attention_mlp", [helper.make_tensor_value_info("x", elem, [1, sequence, width])], outputs, inits
    )
    opset = 20 if activation == "Gelu" else 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model, full_check=True)
    return model


def _names(plan: quantize.Int8Plan) -> list[str]:
    """Return the last two path segments of each weighted node, e.g. ``"fc1/MatMul"``.

    Examples:
        >>> _names(quantize.Int8Plan(weighted=("/a/b/fc1/MatMul",), activations=()))
        ['fc1/MatMul']
    """
    return ["/".join(name.split("/")[-2:]) for name in plan.weighted]


def _outputs_before_and_after_lift(model: Any, feeds: dict[str, np.ndarray]) -> tuple[list[Any], list[Any]]:
    """Run *model* on onnxruntime's CPU provider as given and again after ``quantize._lift_opset`` raised it to 19.

    Args:
        model: A ``ModelProto`` at opset 17; left untouched (the lift runs on a copy).
        feeds: Graph inputs by name.

    Returns:
        The outputs at opset 17 and the outputs at opset 19.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> node = helper.make_node("Relu", ["x"], ["y"])
        >>> graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
        >>> model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        >>> before, after = _outputs_before_and_after_lift(model, {"x": np.array([-1.0, 2.0], np.float32)})
        >>> before[0].tolist(), after[0].tolist()
        ([0.0, 2.0], [0.0, 2.0])
    """
    import onnxruntime

    original, lifted = type(model)(), type(model)()
    original.CopyFrom(model)
    lifted.CopyFrom(model)
    quantize._lift_opset(lifted)
    outputs = []
    for candidate in (original, lifted):
        candidate.ir_version = 9  # the newest IR onnx writes can be newer than the installed onnxruntime reads
        session = onnxruntime.InferenceSession(candidate.SerializeToString(), providers=["CPUExecutionProvider"])
        outputs.append(session.run(None, feeds))
    return outputs[0], outputs[1]
