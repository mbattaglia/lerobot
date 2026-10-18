# Hardened LeRobot

This fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) starts at release `v0.6.1`.
It changes defaults so that nothing reaches the Hugging Face Hub unless you ask.

## Changes

| Area | Upstream `v0.6.1` | This fork |
| --- | --- | --- |
| Hub access | Online | Offline: `import lerobot` sets `HF_HUB_OFFLINE=1` unless it is already set |
| Hugging Face telemetry | On when online | Off: `import lerobot` sets `HF_HUB_DISABLE_TELEMETRY=1` unless it is already set |
| `lerobot-record --dataset.push_to_hub` | `true` | `false` |
| Dataset and policy visibility | Hub default (public for personal accounts) | Private |
| `LeRobotDataset.push_to_hub`, `push_model_to_hub`, checkpoint pushes | Hub default | Private unless `private=False` |
| Upload while offline | Fails after the run | Fails at startup with the fix in the message |
| Async inference and RL actor/learner messages | `pickle`: a peer can run code on the receiver | JSON plus safetensors (`lerobot/utils/safe_serialization.py`): only listed LeRobot types are rebuilt |

Cloud training through `--job.target` still pushes, because the result only exists on the Hub.
That path needs `HF_HUB_OFFLINE=0`.

## Async inference

`policy_server` and `robot_client` run on machines you choose, not on Hugging Face servers.
Upstream they unpickle each other's messages, so either side could run code on the other.
This fork sends JSON plus safetensors instead. The receiver rebuilds only these types:
`RemotePolicyConfig`, `TimedObservation`, `TimedAction`, `PolicyFeature`, and `FeatureType`.

Both sides must run this fork. An upstream client or server fails with
"Not a LeRobot safe-serialization message".

The gRPC channel is still unencrypted and unauthenticated.
A server you connect to still chooses the arm's actions.
To use a remote GPU, bind the server to `127.0.0.1` and reach it through an SSH tunnel:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-host
```

## Install for the SO-101

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --exclude-newer 2026-10-11T00:00:00Z \
  "lerobot[feetech,dataset,pynput-dep] @ git+https://github.com/mbattaglia/lerobot.git@<commit>"
```

Replace `<commit>` with a reviewed commit of this fork.
`feetech` drives the motors, `dataset` records, and `pynput-dep` provides recording keyboard controls.
This set leaves out `rerun-sdk`, `foxglove-sdk`, `wandb`, and `accelerate`.
Add `viz` only if you want `--display_data=true`.

## Models

Download models yourself, at a pinned revision, then run offline:

```bash
HF_HUB_OFFLINE=0 hf download lerobot/<model> --revision <commit>
hf cache ls
```

Some policies also download backbones from other publishers.
For example, SmolVLA uses `HuggingFaceTB/...` and pi0-FAST uses `google/paligemma-...`.
Download those yourself as well. Offline mode turns a missing download into an error.

## Uploading on purpose

```bash
HF_HUB_OFFLINE=0 lerobot-record ... --dataset.push_to_hub=true
```

Uploads stay private unless you also pass `--dataset.private=false`.

## Tests

The test suite sets `HF_HUB_OFFLINE=0` in `tests/conftest.py`, because it downloads fixtures.
`tests/test_offline_defaults.py` checks the new defaults in a clean subprocess.
