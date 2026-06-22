# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Analyze tune-preview cache and propose optimized inference parameters.

Backward-compatible facade — prefer submodules for new code.
"""

from __future__ import annotations

from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.tuning.auto_tune_metrics import analyze_tune_cache
from rfdetr_demo.tuning.auto_tune_proposer import evaluate_auto_tune, propose_parameters
from rfdetr_demo.tuning.auto_tune_types import (
    DEFAULT_PARAMETERS,
    AnomalyFlags,
    AutoTuneEffectiveness,
    CacheQualityMetrics,
    CurrentParameters,
    ProposedParameters,
)

__all__ = [
    "DEFAULT_PARAMETERS",
    "AnomalyFlags",
    "AutoTuneEffectiveness",
    "CacheQualityMetrics",
    "CurrentParameters",
    "ProposedParameters",
    "analyze_tune_cache",
    "evaluate_auto_tune",
    "propose_parameters",
    "run_auto_tune",
]


def run_auto_tune(
    cache: TunePreviewCache,
    *,
    current: CurrentParameters | None = None,
) -> tuple[ProposedParameters, CacheQualityMetrics, AutoTuneEffectiveness]:
    """Full pipeline: analyze cache, propose params, evaluate effectiveness."""
    params = current if current is not None else DEFAULT_PARAMETERS
    metrics = analyze_tune_cache(cache, current=params)
    proposed = propose_parameters(metrics, current=params)
    effectiveness = evaluate_auto_tune(cache, current=params, proposed=proposed)
    return proposed, metrics, effectiveness
