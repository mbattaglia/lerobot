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
"""`lerobot-train --use_torch_compile`: the flags, the in-place compile and its fallbacks.

The compile test uses the `aot_eager` backend: it runs Dynamo and AOTAutograd (the forward and
backward graphs) like `inductor` does, but skips code generation, which takes about a minute for
ACT on CPU. Inductor itself is exercised by the GPU benchmark, not by this CPU suite.
"""

import json
import logging

import draccus
import pytest
import torch
from safetensors.torch import load_file

# Importing the training script pulls in the dataset stack; the fast-test tiers that do not
# install `lerobot[dataset]` skip this module rather than fail collection.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _compile_policy_in_place,
    _first_update_with_compile_fallback,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

TEST_BACKEND = "aot_eager"
IMAGE_KEY = "observation.images.top"


def _torch_compile_usable() -> bool:
    if not hasattr(torch, "compile"):
        return False
    try:
        torch._dynamo.reset()
        return bool(torch.compile(lambda x: x + 1, backend=TEST_BACKEND)(torch.ones(1)).item() == 2)
    except Exception:
        return False
    finally:
        torch._dynamo.reset()


requires_torch_compile = pytest.mark.skipif(
    not _torch_compile_usable(), reason="torch.compile is not usable in this environment"
)


def _tiny_act_policy() -> ACTPolicy:
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        pretrained_backbone_weights=None,
    )
    return ACTPolicy(config)


def _tiny_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 3),
        IMAGE_KEY: torch.rand(batch_size, 3, 32, 32),
        ACTION: torch.randn(batch_size, 4, 3),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


class TestFlags:
    def test_defaults_are_off(self):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
        assert cfg.use_torch_compile is False
        assert cfg.torch_compile_backend == "inductor"
        assert cfg.torch_compile_mode == "default"

    def test_cli_parses_the_flags(self):
        cfg = draccus.parse(
            TrainPipelineConfig,
            args=[
                "--dataset.repo_id=lerobot/dummy",
                "--use_torch_compile=true",
                "--torch_compile_backend=aot_eager",
                "--torch_compile_mode=max-autotune",
            ],
        )
        assert cfg.use_torch_compile is True
        assert cfg.torch_compile_backend == "aot_eager"
        assert cfg.torch_compile_mode == "max-autotune"

    def test_resume_config_can_turn_compile_off(self, tmp_path):
        """A run saved with compile on resumes with it off when the CLI says so, and vice versa."""
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), use_torch_compile=True)
        cfg._save_pretrained(tmp_path)
        config_file = tmp_path / TRAIN_CONFIG_NAME

        assert TrainPipelineConfig.from_pretrained(config_file).use_torch_compile is True
        resumed = TrainPipelineConfig.from_pretrained(config_file, cli_args=["--use_torch_compile=false"])
        assert resumed.use_torch_compile is False

        cfg.use_torch_compile = False
        cfg._save_pretrained(tmp_path)
        resumed = TrainPipelineConfig.from_pretrained(config_file, cli_args=["--use_torch_compile=true"])
        assert resumed.use_torch_compile is True

    def test_config_saved_before_the_flags_existed_still_loads(self, tmp_path):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
        cfg._save_pretrained(tmp_path)
        config_file = tmp_path / TRAIN_CONFIG_NAME
        payload = json.loads(config_file.read_text())
        for key in ("use_torch_compile", "torch_compile_backend", "torch_compile_mode"):
            del payload[key]
        config_file.write_text(json.dumps(payload))

        loaded = TrainPipelineConfig.from_pretrained(config_file)
        assert loaded.use_torch_compile is False
        assert loaded.torch_compile_backend == "inductor"
        assert loaded.torch_compile_mode == "default"


@requires_torch_compile
def test_in_place_compile_keeps_checkpoints_unchanged(tmp_path):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    reference_keys = set(_tiny_act_policy().state_dict())

    policy = _tiny_act_policy()
    policy.train()
    assert _compile_policy_in_place(policy, backend=TEST_BACKEND, mode="default")

    loss, _ = policy(_tiny_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0, "the forward was not compiled"
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)

    assert set(policy.state_dict()) == reference_keys
    policy.save_pretrained(tmp_path / "compiled")
    saved = load_file(tmp_path / "compiled" / "model.safetensors")
    assert not any("_orig_mod" in key for key in saved)

    # Saved with compile on -> loads strictly into an uncompiled policy, with the same weights.
    reloaded = ACTPolicy.from_pretrained(tmp_path / "compiled", strict=True)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value)

    # Saved with compile off -> loads strictly, then compiles and trains.
    _tiny_act_policy().save_pretrained(tmp_path / "eager")
    resumed = ACTPolicy.from_pretrained(tmp_path / "eager", strict=True)
    resumed.train()
    assert _compile_policy_in_place(resumed, backend=TEST_BACKEND, mode="default")
    loss, _ = resumed(_tiny_batch())
    loss.backward()
    assert torch.isfinite(loss)
    torch._dynamo.reset()


def test_bad_backend_warns_and_leaves_the_policy_eager(caplog):
    policy = torch.nn.Linear(2, 2)
    with caplog.at_level(logging.WARNING):
        assert not _compile_policy_in_place(policy, backend="no_such_backend", mode="default")
    assert "training without it" in caplog.text
    assert policy._compiled_call_impl is None


def test_missing_torch_compile_warns(monkeypatch, caplog):
    monkeypatch.delattr(torch, "compile")
    with caplog.at_level(logging.WARNING):
        assert not _compile_policy_in_place(torch.nn.Linear(2, 2), backend="inductor", mode="default")
    assert "not available" in caplog.text


def test_first_step_failure_falls_back_to_eager(caplog):
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    policy._compiled_call_impl = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    calls = []

    def update():
        calls.append(policy._compiled_call_impl is None)
        loss = policy(torch.ones(1, 2)).sum()
        loss.backward()
        return "tracker", None

    with caplog.at_level(logging.WARNING):
        assert _first_update_with_compile_fallback(update, policy, optimizer) == ("tracker", None)
    assert calls == [False, True]
    assert policy._compiled_call_impl is None
    assert "boom" in caplog.text
