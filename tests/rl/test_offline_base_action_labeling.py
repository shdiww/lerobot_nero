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
"""Tests for labeling offline demonstration buffers with base actions (residual TD3)."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.policies.td3_actor.configuration_td3_actor import OBS_BASE_ACTION  # noqa: E402
from lerobot.rl.buffer import ReplayBuffer  # noqa: E402
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD  # noqa: E402


class FakeDataset:
    """Two episodes of 3 and 2 frames, with a per-frame scalar state."""

    def __init__(self):
        # (state value, done) per frame; episode boundaries at index 2 and 4
        self.frames = [
            (0.0, False),
            (1.0, False),
            (2.0, True),
            (3.0, False),
            (4.0, True),
        ]
        self.episodes = [0, 0, 0, 1, 1]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        state, done = self.frames[i]
        return {
            OBS_STATE: torch.tensor([state]),
            ACTION: torch.tensor([0.5]),
            REWARD: torch.tensor(0.0),
            DONE: torch.tensor(done),
            "episode_index": torch.tensor(self.episodes[i]),
        }


class CountingLabeler:
    """Returns per-frame deterministic base actions and counts calls/resets."""

    def __init__(self):
        self.calls = 0
        self.resets = 0

    def __call__(self, sample):
        self.calls += 1
        return torch.full((2,), float(sample[OBS_STATE].item()) * 0.1)

    def reset(self):
        self.resets += 1


def test_transitions_labeled_with_base_actions():
    labeler = CountingLabeler()
    transitions = ReplayBuffer._lerobotdataset_to_transitions(
        dataset=FakeDataset(),
        state_keys=[OBS_STATE, OBS_BASE_ACTION],
        base_action_fn=labeler,
    )

    assert len(transitions) == 5
    # One base-action computation per frame (next-state calls are cached)
    assert labeler.calls == 5
    # One reset at the single episode boundary (frames 2 -> 3)
    assert labeler.resets == 1

    for i, t in enumerate(transitions):
        expected = torch.full((1, 2), float(i) * 0.1)
        assert torch.allclose(t["state"][OBS_BASE_ACTION], expected)

    # Non-terminal next states carry the next frame's base action
    for i in (0, 1, 3):
        expected_next = torch.full((1, 2), float(i + 1) * 0.1)
        assert torch.allclose(transitions[i]["next_state"][OBS_BASE_ACTION], expected_next)

    # Terminal next states are zero-padded and never bootstrapped from
    for i in (2, 4):
        assert torch.all(transitions[i]["next_state"][OBS_BASE_ACTION] == 0.0)

    # The default `next_state is current_state` aliasing must not leak the
    # zero padding into the terminal frame's own state
    assert torch.allclose(transitions[2]["state"][OBS_BASE_ACTION], torch.full((1, 2), 0.2))


def test_base_action_key_requires_labeler():
    with pytest.raises(ValueError, match="base_action_fn"):
        ReplayBuffer._lerobotdataset_to_transitions(
            dataset=FakeDataset(),
            state_keys=[OBS_STATE, OBS_BASE_ACTION],
            base_action_fn=None,
        )


def test_transitions_without_base_action_key_unchanged():
    transitions = ReplayBuffer._lerobotdataset_to_transitions(
        dataset=FakeDataset(),
        state_keys=[OBS_STATE],
        base_action_fn=None,
    )
    assert len(transitions) == 5
    for t in transitions:
        assert OBS_BASE_ACTION not in t["state"]
