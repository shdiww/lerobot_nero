#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for TrainRLServerPipelineConfig dataset-stats population."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

import lerobot.datasets  # noqa: E402
from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.policies.factory import make_policy_config  # noqa: E402
from lerobot.policies.td3_actor import TD3ActorConfig  # noqa: E402
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402


class _FakeMeta:
    def __init__(self, repo_id, root):
        self.stats = {
            "action": {"min": torch.tensor([0.0, -1.0]), "max": torch.tensor([1.0, 2.0])},
            "observation.state": {"min": torch.tensor([0.0]), "max": torch.tensor([1.0])},
        }


def test_validate_populates_policy_dataset_stats_from_dataset(monkeypatch):
    monkeypatch.setattr(lerobot.datasets, "LeRobotDatasetMetadata", _FakeMeta)

    cfg = TrainRLServerPipelineConfig()
    cfg.policy = make_policy_config("td3_actor")
    cfg.policy.push_to_hub = False
    assert isinstance(cfg.policy, TD3ActorConfig)
    cfg.dataset = DatasetConfig(repo_id="fake/demos")

    cfg.validate()

    assert cfg.policy.dataset_stats["action"]["min"] == [0.0, -1.0]
    assert cfg.policy.dataset_stats["action"]["max"] == [1.0, 2.0]
    assert cfg.policy.dataset_stats["observation.state"]["max"] == [1.0]
    # Algorithm config is wired by validate as well
    assert cfg.algorithm is not None
    assert cfg.algorithm.policy_config is cfg.policy


def test_validate_without_dataset_leaves_policy_stats_untouched(monkeypatch):
    def _unexpected_meta(repo_id, root):
        raise AssertionError("dataset metadata should not be loaded without a dataset")

    monkeypatch.setattr(
        lerobot.datasets,
        "LeRobotDatasetMetadata",
        lambda repo_id, root: _unexpected_meta(repo_id, root),
    )

    cfg = TrainRLServerPipelineConfig()
    cfg.policy = make_policy_config("td3_actor")
    cfg.policy.push_to_hub = False
    cfg.validate()
    # Placeholder stats from the policy config remain
    assert cfg.policy.dataset_stats is not None
