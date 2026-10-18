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
"""Messages from a network peer are decoded without pickle."""

import json
import os
import pickle  # nosec B403 - used only to build hostile payloads
import struct
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.safe_serialization import dumps, loads


@dataclass
class _Msg:
    name: str
    feature: PolicyFeature
    frames: dict


def test_round_trip_of_nested_values():
    obj = {
        "none": None,
        "flag": True,
        "count": 3,
        "ratio": 0.5,
        "text": "héllo",
        "raw": b"\x00\x01",
        "items": [1, (2, 3), {"k": "v"}],
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "image": np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8),
        "scalar": np.float32(1.5),
    }
    out = loads(dumps(obj))
    assert out["none"] is None and out["flag"] is True and out["count"] == 3
    assert out["text"] == "héllo" and out["raw"] == b"\x00\x01"
    assert out["items"] == [1, (2, 3), {"k": "v"}]
    torch.testing.assert_close(out["tensor"], obj["tensor"])
    assert isinstance(out["image"], np.ndarray) and out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["image"], obj["image"])
    assert out["scalar"] == 1.5


def test_allowed_dataclasses_and_enums_round_trip():
    msg = _Msg("cam", PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 5)), {"t": torch.ones(2)})
    out = loads(dumps(msg), allowed=(_Msg, PolicyFeature, FeatureType))
    assert out.name == "cam" and out.feature.type is FeatureType.VISUAL and out.feature.shape == (3, 4, 5)
    torch.testing.assert_close(out.frames["t"], torch.ones(2))


def test_dataclass_not_in_allowed_list_is_refused():
    data = dumps(PolicyFeature(type=FeatureType.STATE, shape=(6,)))
    with pytest.raises(ValueError, match="not allowed"):
        loads(data, allowed=(FeatureType,))


def test_enum_not_in_allowed_list_is_refused():
    with pytest.raises(ValueError, match="not allowed"):
        loads(dumps(FeatureType.ACTION))


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_pickle_payload_is_rejected_without_running():
    payload = pickle.dumps(_Exploit())
    with pytest.raises(ValueError, match="Not a LeRobot"):
        loads(payload)


def _raw_message(tree) -> bytes:
    header = json.dumps(tree).encode()
    return b"LRSS1" + struct.pack("<Q", len(header)) + header


@pytest.mark.parametrize(
    "tree",
    [
        {"__enum__": "posix.system", "value": "echo pwned"},
        {"__dataclass__": "subprocess.Popen", "fields": {"args": "echo pwned"}},
        {"__dataclass__": "lerobot.configs.types.FeatureType", "fields": {}},
    ],
)
def test_message_cannot_name_other_callables(tree):
    with pytest.raises(ValueError):
        loads(_raw_message(tree), allowed=(FeatureType,))


@pytest.mark.parametrize(
    "data", [b"", b"LRSS1", b"LRSS1" + b"\xff" * 8, b"LRSS1\x02\x00\x00\x00\x00\x00\x00\x00{]"]
)
def test_malformed_input_raises_value_error(data):
    with pytest.raises(ValueError):
        loads(data)


def test_unsupported_types_are_refused_on_encode():
    with pytest.raises(TypeError):
        dumps({1: "int key"})
    with pytest.raises(TypeError):
        dumps(object())
    with pytest.raises(TypeError):
        dumps(np.array([object()], dtype=object))
