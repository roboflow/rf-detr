# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unified tune-preview analysis and parameter optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from rfdetr_demo.inference.tune_cache import TunePreviewCache
from rfdetr_demo.tuning.auto_tune import (
    DEFAULT_PARAMETERS,
    AutoTuneEffectiveness,
    CacheQualityMetrics,
    CurrentParameters,
    ProposedParameters,
    analyze_tune_cache,
    evaluate_auto_tune,
    propose_parameters,
)


@dataclass(frozen=True)
class TunePipelineResult:
    """Output of the tune-preview optimization pipeline."""

    metrics: CacheQualityMetrics
    proposed: ProposedParameters
    effectiveness: AutoTuneEffectiveness


def run_tune_pipeline(
    cache: TunePreviewCache,
    current: CurrentParameters | None = None,
) -> TunePipelineResult:
    """Analyze tune cache, propose parameters, and evaluate effectiveness."""
    params = current or DEFAULT_PARAMETERS
    metrics = analyze_tune_cache(cache, current=params)
    proposed = propose_parameters(metrics, current=params)
    effectiveness = evaluate_auto_tune(cache, current=params, proposed=proposed)
    return TunePipelineResult(metrics=metrics, proposed=proposed, effectiveness=effectiveness)
