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
"""`lerobot-train --use_tf32` and `--use_channels_last`: the flags, the settings and checkpoints.

channels_last only changes how the conv weights are laid out in memory, never their values, so a
checkpoint trained with it must hold the same keys and values as the live policy, in the standard
layout, and load into a policy that never used the flag.
"""

import json
import logging
import shutil

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
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.scripts.lerobot_train import (  # noqa: E402
    _compile_policy_in_place,
    _convert_to_channels_last,
    _enable_tf32,
    _match_optimizer_state_memory_format,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
)

IMAGE_KEY = "observation.images.top"
DATASET_REPO_ID = "dummy/channels_last"
TEST_BACKEND = "aot_eager"
NEW_FLAGS = ("use_tf32", "use_channels_last")


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
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


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
    generator = torch.Generator().manual_seed(1)
    return {
        OBS_STATE: torch.randn(batch_size, 3, generator=generator),
        IMAGE_KEY: torch.rand(batch_size, 3, 32, 32, generator=generator),
        ACTION: torch.randn(batch_size, 4, 3, generator=generator),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def _conv_weights(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.weight for name, module in policy.named_modules() if isinstance(module, torch.nn.Conv2d)
    }


def _is_channels_last(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous(memory_format=torch.channels_last)


# ---------------------------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------------------------


class TestFlags:
    def test_defaults_are_off(self):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
        assert cfg.use_tf32 is False
        assert cfg.use_channels_last is False

    def test_cli_parses_the_flags(self):
        cfg = draccus.parse(
            TrainPipelineConfig,
            args=["--dataset.repo_id=lerobot/dummy", "--use_tf32=true", "--use_channels_last=true"],
        )
        assert cfg.use_tf32 is True
        assert cfg.use_channels_last is True

        cfg = draccus.parse(
            TrainPipelineConfig,
            args=["--dataset.repo_id=lerobot/dummy", "--use_tf32=false", "--use_channels_last=false"],
        )
        assert cfg.use_tf32 is False
        assert cfg.use_channels_last is False

    def test_resume_config_can_turn_the_flags_on_and_off(self, tmp_path):
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="lerobot/dummy"), use_tf32=True, use_channels_last=True
        )
        cfg._save_pretrained(tmp_path)
        config_file = tmp_path / TRAIN_CONFIG_NAME

        loaded = TrainPipelineConfig.from_pretrained(config_file)
        assert (loaded.use_tf32, loaded.use_channels_last) == (True, True)
        resumed = TrainPipelineConfig.from_pretrained(
            config_file, cli_args=["--use_tf32=false", "--use_channels_last=false"]
        )
        assert (resumed.use_tf32, resumed.use_channels_last) == (False, False)

    def test_config_saved_before_the_flags_existed_still_loads(self, tmp_path):
        cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"))
        cfg._save_pretrained(tmp_path)
        config_file = tmp_path / TRAIN_CONFIG_NAME
        payload = json.loads(config_file.read_text())
        for key in NEW_FLAGS:
            del payload[key]
        config_file.write_text(json.dumps(payload))

        loaded = TrainPipelineConfig.from_pretrained(config_file)
        assert loaded.use_tf32 is False
        assert loaded.use_channels_last is False


# ---------------------------------------------------------------------------------------------
# --use_tf32
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def tf32_off(monkeypatch):
    """Start from TF32 off and restore the process-wide switches afterwards."""
    precision = torch.get_float32_matmul_precision()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def test_use_tf32_sets_the_cuda_backend_switches(tf32_off, caplog):
    # The switches are process-wide settings, so this runs without a GPU too.
    with caplog.at_level(logging.INFO):
        _enable_tf32(torch.device("cuda"))
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"
    assert "TF32 enabled" in caplog.text
    assert "float32_matmul_precision=high" in caplog.text


def test_use_tf32_does_nothing_off_cuda(tf32_off, caplog):
    with caplog.at_level(logging.INFO):
        _enable_tf32(torch.device("cpu"))
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert "does nothing" in caplog.text


@requires_cuda
def test_use_tf32_on_a_cuda_device_runs_tf32_matmuls(tf32_off):
    _enable_tf32(torch.device("cuda"))
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    exact = (a.double() @ b.double()).float()
    # TF32 keeps a 10-bit mantissa: close to, but not exactly, the float32 result.
    torch.testing.assert_close(a @ b, exact, atol=0.5, rtol=1e-2)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


# ---------------------------------------------------------------------------------------------
# --use_channels_last: the policy and its checkpoint
# ---------------------------------------------------------------------------------------------


def test_channels_last_checkpoint_loads_into_a_policy_without_it(tmp_path):
    policy = _tiny_act_policy()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    assert _convert_to_channels_last(policy) > 0

    # Only the layout changes: same keys and values, and the kxk conv weights are no longer
    # contiguous in the default layout (which is what safetensors would refuse).
    assert policy.state_dict().keys() == before.keys()
    for key, value in policy.state_dict().items():
        assert torch.equal(value, before[key]), key
    weights = _conv_weights(policy)
    assert all(_is_channels_last(weight) for weight in weights.values())
    assert not weights["model.backbone.conv1"].is_contiguous()

    # One forward and backward pass; the gradients follow the weights' layout.
    policy.train()
    loss, _ = policy(_tiny_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert _is_channels_last(policy.model.backbone.conv1.weight.grad)
    torch.optim.SGD(policy.parameters(), lr=0.1).step()  # make the saved weights non-trivial

    policy.save_pretrained(tmp_path / "model")
    saved = load_file(tmp_path / "model" / "model.safetensors")
    assert saved.keys() == policy.state_dict().keys()
    for key, value in policy.state_dict().items():
        assert saved[key].is_contiguous(), key
        assert torch.equal(saved[key], value), key

    # A fresh policy that never used the flag loads the checkpoint strictly, with equal weights.
    fresh = ACTPolicy.from_pretrained(tmp_path / "model", strict=True)
    assert fresh.state_dict().keys() == saved.keys()
    for key, value in fresh.state_dict().items():
        assert value.is_contiguous(), key
        assert torch.equal(value, saved[key]), key

    policy.eval()
    fresh.eval()
    with torch.no_grad():
        batch = _tiny_batch()
        torch.testing.assert_close(
            fresh.predict_action_chunk(dict(batch)), policy.predict_action_chunk(dict(batch))
        )
        torch.testing.assert_close(fresh(dict(batch))[0], policy(dict(batch))[0])


def test_channels_last_hands_channels_last_images_to_the_backbone():
    policy = _tiny_act_policy()
    _convert_to_channels_last(policy)
    seen = {}

    def record(module, args):
        seen["first_conv_input"] = args[0]

    # Registered after the conversion hook, so it sees what the conv actually receives.
    policy.model.backbone.conv1.register_forward_pre_hook(record)
    policy.train()
    batch = _tiny_batch()
    assert not _is_channels_last(batch[IMAGE_KEY])
    policy(batch)
    assert _is_channels_last(seen["first_conv_input"])


def test_channels_last_with_autocast():
    policy = _tiny_act_policy()
    _convert_to_channels_last(policy)
    policy.train()
    outputs = {}
    policy.model.backbone.conv1.register_forward_hook(lambda m, a, out: outputs.update(conv1=out))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, _ = policy(_tiny_batch())
    loss.backward()
    assert torch.isfinite(loss)
    # autocast runs the conv in low precision and keeps the channels_last layout.
    assert outputs["conv1"].dtype == torch.bfloat16
    assert _is_channels_last(outputs["conv1"])
    assert not outputs["conv1"].is_contiguous()
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)


@requires_torch_compile
def test_channels_last_with_compile(tmp_path):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    # Converted before compile, as train() does.
    policy = _tiny_act_policy()
    _convert_to_channels_last(policy)
    policy.train()
    assert _compile_policy_in_place(policy, backend=TEST_BACKEND, mode="default")
    loss, _ = policy(_tiny_batch())
    loss.backward()
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0, "the forward was not compiled"
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in policy.parameters() if p.requires_grad)
    assert _is_channels_last(policy.model.backbone.conv1.weight.grad)

    policy.save_pretrained(tmp_path / "compiled")
    fresh = ACTPolicy.from_pretrained(tmp_path / "compiled", strict=True)
    for key, value in policy.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value), key

    # Without dropout, the compiled channels_last policy gives the output of the plain eager one.
    policy.eval()
    fresh.eval()
    with torch.no_grad():
        batch = _tiny_batch()
        torch.testing.assert_close(policy(dict(batch))[0], fresh(dict(batch))[0])
    torch._dynamo.reset()


def test_channels_last_does_nothing_without_conv_layers():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    assert _convert_to_channels_last(model) == 0


def test_optimizer_state_saves_contiguous_and_resumes_in_channels_last(tmp_path):
    policy = _tiny_act_policy()
    _convert_to_channels_last(policy)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    loss, _ = policy(_tiny_batch())
    loss.backward()
    optimizer.step()
    conv1 = policy.model.backbone.conv1.weight
    assert not optimizer.state[conv1]["exp_avg"].is_contiguous()

    # safetensors refuses non-contiguous tensors: the state must be written in the standard layout.
    save_optimizer_state(optimizer, tmp_path)

    resumed_policy = _tiny_act_policy()
    _convert_to_channels_last(resumed_policy)
    resumed = torch.optim.AdamW(resumed_policy.parameters(), lr=1e-3)
    load_optimizer_state(resumed, tmp_path)
    _match_optimizer_state_memory_format(resumed)

    resumed_conv1 = resumed_policy.model.backbone.conv1.weight
    for key in ("exp_avg", "exp_avg_sq"):
        assert _is_channels_last(resumed.state[resumed_conv1][key])
    for original, loaded in zip(policy.parameters(), resumed_policy.parameters(), strict=True):
        for key, value in optimizer.state[original].items():
            assert torch.equal(resumed.state[loaded][key], value), key

    # The same state also loads into an optimizer of a policy without channels_last.
    plain_policy = _tiny_act_policy()
    plain = torch.optim.AdamW(plain_policy.parameters(), lr=1e-3)
    load_optimizer_state(plain, tmp_path)
    plain_conv1 = plain_policy.model.backbone.conv1.weight
    assert torch.equal(plain.state[plain_conv1]["exp_avg"], optimizer.state[conv1]["exp_avg"])


# ---------------------------------------------------------------------------------------------
# train(): end to end, with resume
# ---------------------------------------------------------------------------------------------


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


def _train_config(root, output_dir, steps: int, **overrides) -> TrainPipelineConfig:
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
        steps=steps,
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
    return cfg


def _weights(output_dir, step: int) -> dict[str, torch.Tensor]:
    return load_file(output_dir / "checkpoints" / f"{step:06d}" / PRETRAINED_MODEL_DIR / "model.safetensors")


def _assert_same_weights(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].is_contiguous(), key
        # channels_last convs may sum in another order: equal up to float rounding.
        torch.testing.assert_close(actual[key], value, atol=1e-5, rtol=1e-4, msg=key)


@pytest.fixture
def dataset_root(tmp_path):
    pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")
    return _make_dataset(tmp_path)


@pytest.fixture
def reset_accelerate():
    yield
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def _resume(root, run_dir, copy_dir, steps: int, **overrides) -> dict[str, torch.Tensor]:
    """Resume a copy of `run_dir` from its step-2 checkpoint up to `steps`."""
    shutil.copytree(run_dir, copy_dir, symlinks=True)  # keep `checkpoints/last` a symlink
    cfg = _train_config(root, copy_dir, steps=steps, **overrides)
    cfg.resume = True
    cfg.checkpoint_path = copy_dir / "checkpoints" / "000002"
    lerobot_train.train(cfg)
    return _weights(copy_dir, steps)


def test_train_with_channels_last_matches_the_default_and_resumes_either_way(
    dataset_root, tmp_path, monkeypatch, reset_accelerate
):
    converted, matched = [], []
    monkeypatch.setattr(
        lerobot_train,
        "_convert_to_channels_last",
        lambda policy, _real=lerobot_train._convert_to_channels_last: converted.append(1) or _real(policy),
    )
    monkeypatch.setattr(
        lerobot_train,
        "_match_optimizer_state_memory_format",
        lambda opt, _real=lerobot_train._match_optimizer_state_memory_format: matched.append(1) or _real(opt),
    )

    plain_dir, cl_dir = tmp_path / "plain", tmp_path / "channels_last"
    lerobot_train.train(_train_config(dataset_root, plain_dir, steps=2))
    lerobot_train.train(_train_config(dataset_root, cl_dir, steps=2, use_channels_last=True))
    assert converted == [1]
    _assert_same_weights(_weights(cl_dir, 2), _weights(plain_dir, 2))
    # The optimizer state was saved (safetensors would refuse non-contiguous tensors).
    assert (cl_dir / "checkpoints" / "000002" / TRAINING_STATE_DIR / "optimizer_state.safetensors").is_file()

    # Resume a channels_last run without the flag, and a default run with it.
    cl_then_plain = _resume(dataset_root, cl_dir, tmp_path / "cl_then_plain", steps=4)
    plain_then_cl = _resume(
        dataset_root, plain_dir, tmp_path / "plain_then_cl", steps=4, use_channels_last=True
    )
    plain_then_plain = _resume(dataset_root, plain_dir, tmp_path / "plain_then_plain", steps=4)
    assert converted == [1, 1]
    assert matched == [1]
    _assert_same_weights(cl_then_plain, plain_then_plain)
    _assert_same_weights(plain_then_cl, plain_then_plain)


def test_train_with_use_tf32_on_cpu_logs_that_it_does_nothing(
    dataset_root, tmp_path, capfd, tf32_off, reset_accelerate
):
    # train() sets up its own log handlers, so read what reaches the terminal.
    lerobot_train.train(_train_config(dataset_root, tmp_path / "run", steps=2, use_tf32=True))
    out, err = capfd.readouterr()
    assert "--use_tf32 only applies to CUDA" in out + err
    assert "TF32 enabled" not in out + err
    assert torch.backends.cudnn.allow_tf32 is False
