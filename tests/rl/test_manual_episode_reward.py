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
"""Tests for manual (terminal) episode-end reward labeling, ResFiT-style."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.rl.actor import (  # noqa: E402
    apply_manual_episode_reward,
    prompt_episode_reward,
)


def _fake_transitions(n: int = 3) -> list[dict]:
    return [{"state": {}, "action": torch.zeros(1), "reward": 0.0} for _ in range(n)]


def test_apply_manual_episode_reward_sets_terminal_transition():
    transitions = _fake_transitions()
    apply_manual_episode_reward(transitions, 1.0)
    assert transitions[-1]["reward"] == 1.0
    assert all(t["reward"] == 0.0 for t in transitions[:-1])


def test_apply_manual_episode_reward_empty_list_is_noop():
    apply_manual_episode_reward([], 1.0)  # must not raise


def test_prompt_episode_reward_accepts_common_labels():
    assert prompt_episode_reward(input_fn=lambda _: "1") == 1.0
    assert prompt_episode_reward(input_fn=lambda _: "y") == 1.0
    assert prompt_episode_reward(input_fn=lambda _: "YES") == 1.0
    assert prompt_episode_reward(input_fn=lambda _: "0") == 0.0
    assert prompt_episode_reward(input_fn=lambda _: " no ") == 0.0


def test_prompt_episode_reward_retries_on_invalid_input():
    answers = iter(["maybe", "", "1"])
    assert prompt_episode_reward(input_fn=lambda _: next(answers)) == 1.0
