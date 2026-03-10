# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Deprecated: use ``rfdetr.utilities`` instead.

ModelEma will move to ``rfdetr.training.model_ema`` in Phase 8.
"""

import warnings

warnings.warn(
    "rfdetr.util.utils is deprecated; use rfdetr.utilities instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations.
from rfdetr.utilities.reproducibility import seed_all  # noqa: E402, F401
from rfdetr.utilities.state_dict import clean_state_dict  # noqa: E402, F401

# ModelEma, BestMetricSingle, BestMetricHolder remain here until Phase 8
# when they move to rfdetr.training.model_ema.
import json  # noqa: E402
import math  # noqa: E402
from collections import OrderedDict  # noqa: E402
from copy import deepcopy  # noqa: E402
from typing import Any, Callable, Dict, Optional, Union  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


class ModelEma(torch.nn.Module):
    """EMA Model"""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9997,
        tau: float = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.tau = tau
        self.updates = 1
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _get_decay(self) -> float:
        if self.tau == 0:
            decay = self.decay
        else:
            decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        return decay

    def _update(
        self,
        model: torch.nn.Module,
        update_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: torch.nn.Module) -> None:
        decay = self._get_decay()
        self._update(model, update_fn=lambda e, m: decay * e + (1.0 - decay) * m)
        self.updates += 1

    def set(self, model: torch.nn.Module) -> None:
        self._update(model, update_fn=lambda e, m: m)


class BestMetricSingle:
    def __init__(self, init_res: float = 0.0, better: str = "large") -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ["large", "small"]

    def isbetter(self, new_res: float, old_res: float) -> bool:
        if self.better == "large":
            return new_res > old_res
        elif self.better == "small":
            return new_res < old_res
        else:
            raise ValueError(f"Unexpected value for 'better': {self.better!r}")

    def update(self, new_res: float, ep: int) -> bool:
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> Dict[str, Union[float, int]]:
        return {
            "best_res": self.best_res,
            "best_ep": self.best_ep,
        }


class BestMetricHolder:
    def __init__(self, init_res: float = 0.0, better: str = "large", use_ema: bool = False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res: float, epoch: int, is_ema: bool = False) -> bool:
        """Return if the results is the best."""
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self) -> Dict[str, Union[float, int]]:
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f"all_{k}": v for k, v in self.best_all.summary().items()})
        res.update({f"regular_{k}": v for k, v in self.best_regular.summary().items()})
        res.update({f"ema_{k}": v for k, v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()


__all__ = [
    "seed_all",
    "clean_state_dict",
    "ModelEma",
    "BestMetricSingle",
    "BestMetricHolder",
]
