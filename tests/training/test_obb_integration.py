# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRBaseConfig, TrainConfig


class TestOrientedNamespace:
    def test_oriented_false_by_default(self) -> None:
        mc = RFDETRBaseConfig(pretrain_weights=None)
        tc = TrainConfig(dataset_dir="/tmp/fake")
        ns = _namespace_from_configs(mc, tc)
        assert ns.oriented is False

    def test_oriented_forwarded_to_namespace(self) -> None:
        mc = RFDETRBaseConfig(pretrain_weights=None, oriented=True)
        tc = TrainConfig(dataset_dir="/tmp/fake")
        ns = _namespace_from_configs(mc, tc)
        assert ns.oriented is True


class TestOrientedConfig:
    def test_oriented_default_false(self) -> None:
        mc = RFDETRBaseConfig(pretrain_weights=None)
        assert mc.oriented is False

    def test_oriented_can_be_set(self) -> None:
        mc = RFDETRBaseConfig(pretrain_weights=None, oriented=True)
        assert mc.oriented is True

    def test_dota_dataset_file_accepted(self) -> None:
        tc = TrainConfig(dataset_dir="/tmp/fake", dataset_file="dota")
        assert tc.dataset_file == "dota"
