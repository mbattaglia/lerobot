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
"""Hub access is opt-in: offline by default, no uploads by default, private when uploading."""

import os
import subprocess
import sys

import draccus
import pytest
from huggingface_hub import constants

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.hub import require_hub_online


def _offline_value_after_import(env_value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")}
    if env_value is not None:
        env["HF_HUB_OFFLINE"] = env_value
    code = "import lerobot, os, huggingface_hub.constants as c; print(os.environ['HF_HUB_OFFLINE'], c.is_offline_mode())"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_import_turns_hub_offline_by_default():
    assert _offline_value_after_import(None) == "1 True"


def test_explicit_opt_in_is_respected():
    assert _offline_value_after_import("0") == "0 False"


def test_import_disables_hub_telemetry():
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_DISABLE_TELEMETRY"}
    code = "import lerobot, huggingface_hub.constants as c; print(c.HF_HUB_DISABLE_TELEMETRY)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def test_dataset_record_defaults_stay_local_and_private():
    cfg = DatasetRecordConfig()
    assert cfg.push_to_hub is False
    assert cfg.private is True


def test_policy_defaults_stay_local_and_private():
    cfg = ACTConfig()
    assert cfg.push_to_hub is False
    assert cfg.private is True


def test_require_hub_online_raises_when_offline(monkeypatch):
    monkeypatch.setattr(constants, "is_offline_mode", lambda: True)
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE=0"):
        require_hub_online("upload")


def test_require_hub_online_passes_when_online(monkeypatch):
    monkeypatch.setattr(constants, "is_offline_mode", lambda: False)
    require_hub_online("upload")


def test_local_training_push_fails_fast_when_offline(monkeypatch):
    monkeypatch.setattr(constants, "is_offline_mode", lambda: True)
    cfg = draccus.parse(
        TrainPipelineConfig,
        args=[
            "--dataset.repo_id=u/d",
            "--policy.type=act",
            "--policy.push_to_hub=true",
            "--policy.repo_id=u/m",
        ],
    )
    with pytest.raises(ValueError, match="HF_HUB_OFFLINE=0"):
        cfg.validate()
