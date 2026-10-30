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
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

# Importing the training script pulls in the dataset stack; the fast-test tiers that do not
# install `lerobot[dataset]` skip this module rather than fail collection.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _compile_policy_in_place,
    _first_update_with_compile_fallback,
    update_policy,
)
from lerobot.utils.constants import ACTION, OBS_STATE, PRETRAINED_MODEL_DIR  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402

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


def test_already_compiled_policy_is_not_compiled_again():
    policy = torch.nn.Linear(2, 2)
    policy.compile(backend="eager")
    compiled_call = policy._compiled_call_impl
    assert not _compile_policy_in_place(policy, backend="eager", mode="default")
    assert policy._compiled_call_impl is compiled_call
    assert not _compile_policy_in_place(torch.compile(torch.nn.Linear(2, 2)), backend="eager", mode="default")


# --- first-step fallback, against a real (CPU) accelerate stack ------------------------------------


class _FailInBackward(torch.autograd.Function):
    """Identity forward; its backward fails like a lazily lowered backward graph would."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("backward lowering failed")


class _TwoLayerPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin1 = torch.nn.Linear(3, 4)
        self.lin2 = torch.nn.Linear(4, 1)

    def forward(self, batch, fail_in_backward: bool = False):
        hidden = self.lin1(batch["x"])
        if fail_in_backward:
            # lin2's backward runs first and leaves its gradients behind; lin1 never gets any.
            hidden = _FailInBackward.apply(hidden)
        loss = torch.nn.functional.mse_loss(self.lin2(hidden), batch["y"])
        return loss, {}


@pytest.fixture
def make_accelerator():
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState

    yield lambda steps: Accelerator(cpu=True, gradient_accumulation_steps=steps)
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def _tracker() -> MetricsTracker:
    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
    }
    return MetricsTracker(1, 1, 1, metrics, initial_step=0)


def _batches(n: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return [
        {"x": torch.randn(2, 3, generator=generator), "y": torch.randn(2, 1, generator=generator)}
        for _ in range(n)
    ]


def test_first_step_backward_failure_under_gradient_accumulation_matches_eager(make_accelerator, caplog):
    """A failed attempt must leave neither partial gradients nor a shifted accumulation counter."""
    batches = _batches(4)

    # Eager reference: two optimizer updates of two micro-batches each.
    reference = _TwoLayerPolicy()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.5)
    for pair in (batches[:2], batches[2:]):
        for batch in pair:
            (reference(batch)[0] / 2).backward()
        reference_optimizer.step()
        reference_optimizer.zero_grad()

    accelerator = make_accelerator(2)
    policy = _TwoLayerPolicy()
    policy, optimizer = accelerator.prepare(policy, torch.optim.SGD(policy.parameters(), lr=0.5))
    policy._compiled_call_impl = lambda batch: policy._call_impl(batch, fail_in_backward=True)
    initial = {name: p.detach().clone() for name, p in policy.named_parameters()}

    def update(batch):
        return update_policy(_tracker(), policy, batch, optimizer, 0.0, accelerator=accelerator)

    with caplog.at_level(logging.WARNING):
        _first_update_with_compile_fallback(lambda: update(batches[0]), policy, optimizer, accelerator)
    assert "RuntimeError: backward lowering failed" in caplog.text
    assert policy._compiled_call_impl is None
    # The first micro-batch must not sync: the update lands after the second, as in eager mode.
    assert not accelerator.sync_gradients
    for name, p in policy.named_parameters():
        torch.testing.assert_close(p, initial[name])

    for batch in batches[1:]:
        update(batch)
    for (name, p), q in zip(policy.named_parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q, msg=name)


def test_error_after_the_optimizer_step_is_not_retried(make_accelerator):
    accelerator = make_accelerator(1)
    policy = _TwoLayerPolicy()
    optimizer = accelerator.prepare(torch.optim.SGD(policy.parameters(), lr=0.5))
    compiled_call = policy._call_impl
    policy._compiled_call_impl = compiled_call
    calls = []

    def update():
        calls.append(1)
        update_policy(_tracker(), policy, _batches(1)[0], optimizer, 0.0, accelerator=accelerator)
        raise RuntimeError("after the step")

    with pytest.raises(RuntimeError, match="after the step"):
        _first_update_with_compile_fallback(update, policy, optimizer, accelerator)
    assert calls == [1]
    assert policy._compiled_call_impl is compiled_call


def test_first_step_lowers_the_backward_graph_during_the_forward(make_accelerator):
    """So a backend failure surfaces before DDP's reducer starts reducing gradient buckets."""
    import torch._functorch.config as functorch_config

    accelerator = make_accelerator(1)
    policy = torch.nn.Linear(2, 2)
    optimizer = accelerator.prepare(torch.optim.SGD(policy.parameters(), lr=0.1))
    seen = []

    def update():
        seen.append(functorch_config.force_non_lazy_backward_lowering)
        return _tracker(), None

    _first_update_with_compile_fallback(update, policy, optimizer, accelerator)
    assert seen == [True]
    assert functorch_config.force_non_lazy_backward_lowering is False


# --- end to end through train() --------------------------------------------------------------------

DATASET_REPO_ID = "dummy/compile"
FAILING_BACKEND = "lerobot_test_failing_backend"


@torch._dynamo.register_backend(name=FAILING_BACKEND)
def _failing_backend(gm, example_inputs):
    raise RuntimeError("this backend always fails")


def _make_dataset(tmp_path):
    features = {
        ACTION: {"dtype": "float32", "shape": (3,), "names": None},
        OBS_STATE: {"dtype": "float32", "shape": (3,), "names": None},
        IMAGE_KEY: {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channel"]},
    }
    root = tmp_path / "_dataset"
    dataset = LeRobotDataset.create(repo_id=DATASET_REPO_ID, fps=30, features=features, root=root)
    rng = np.random.default_rng(0)
    for _ in range(2):
        for _ in range(10):
            dataset.add_frame(
                {
                    ACTION: rng.standard_normal(3).astype(np.float32),
                    OBS_STATE: rng.standard_normal(3).astype(np.float32),
                    IMAGE_KEY: rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
                    "task": "task",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return root


def _train(root, output_dir, **overrides) -> dict[str, torch.Tensor]:
    from lerobot.policies.factory import make_policy_config

    policy_config = make_policy_config(
        "act",
        device="cpu",
        push_to_hub=False,
        chunk_size=4,
        n_action_steps=4,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        dropout=0.0,
        pretrained_backbone_weights=None,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DATASET_REPO_ID, root=str(root)),
        policy=policy_config,
        output_dir=output_dir,
        steps=2,
        batch_size=2,
        num_workers=0,
        seed=42,
        log_freq=0,
        env_eval_freq=0,
        save_freq=2,
        **overrides,
    )
    cfg.optimizer = policy_config.get_optimizer_preset()
    cfg.scheduler = policy_config.get_scheduler_preset()
    # The config is built in-process, so skip the CLI-oriented validation.
    cfg.validate = lambda: None
    lerobot_train.train(cfg)
    return load_file(output_dir / "checkpoints" / "000002" / PRETRAINED_MODEL_DIR / "model.safetensors")


@pytest.fixture
def dataset_root(tmp_path):
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    return _make_dataset(tmp_path)


@requires_torch_compile
def test_train_with_compile_matches_eager(dataset_root, tmp_path, make_accelerator):
    """aot_eager reproduces eager numerics exactly, so the two runs must save the same weights."""
    eager = _train(dataset_root, tmp_path / "eager")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = _train(
        dataset_root, tmp_path / "compiled", use_torch_compile=True, torch_compile_backend=TEST_BACKEND
    )
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0, "train() did not compile"
    assert set(compiled) == set(eager)
    for key, value in eager.items():
        torch.testing.assert_close(compiled[key], value, msg=key)
    torch._dynamo.reset()


@requires_torch_compile
def test_train_falls_back_to_eager_when_compile_fails(dataset_root, tmp_path, monkeypatch, make_accelerator):
    disabled = []
    monkeypatch.setattr(
        lerobot_train,
        "_disable_torch_compile",
        lambda policy, _real=lerobot_train._disable_torch_compile: (disabled.append(policy), _real(policy)),
    )
    eager = _train(dataset_root, tmp_path / "eager")
    torch._dynamo.reset()
    fallback = _train(
        dataset_root, tmp_path / "fallback", use_torch_compile=True, torch_compile_backend=FAILING_BACKEND
    )
    assert len(disabled) == 1
    for key, value in eager.items():
        torch.testing.assert_close(fallback[key], value, msg=key)
    torch._dynamo.reset()
