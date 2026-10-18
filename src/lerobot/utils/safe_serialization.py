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
"""Pickle-free serialization for messages sent between processes or machines.

``pickle.loads`` runs code chosen by whoever produced the bytes, so it must never
read data from a network peer. This format stores the structure as JSON and
tensors/arrays as safetensors. Decoding rebuilds only the dataclasses and enums
the caller lists in ``allowed``; a malicious peer can send bad values, not code.

Supported values: None, bool, int, float, str, bytes, list, tuple, dict with str
keys, torch.Tensor, numpy.ndarray, numpy scalars (decoded as Python numbers),
and allowed dataclasses and enums.
"""

import base64
import dataclasses
import enum
import json
import struct
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as load_tensors, save as save_tensors

_MAGIC = b"LRSS1"
_LENGTH = struct.Struct("<Q")


def _type_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def dumps(obj: Any) -> bytes:
    """Serialize ``obj`` without pickle."""
    tensors: dict[str, torch.Tensor] = {}

    def add_tensor(tensor: torch.Tensor) -> str:
        key = str(len(tensors))
        # clone: safetensors rejects tensors that share storage
        tensors[key] = tensor.detach().cpu().contiguous().clone()
        return key

    def encode(value: Any) -> Any:
        # Before the primitives: str/int enums are also str/int instances.
        if isinstance(value, enum.Enum):
            return {"__enum__": _type_name(type(value)), "value": encode(value.value)}
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return {"__tensor__": add_tensor(value)}
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise TypeError("Cannot serialize numpy arrays of Python objects")
            return {"__ndarray__": add_tensor(torch.from_numpy(np.array(value, copy=True, order="C")))}
        if isinstance(value, bytes):
            return {"__bytes__": base64.b64encode(value).decode("ascii")}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value) if f.init}
            return {"__dataclass__": _type_name(type(value)), "fields": fields}
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise TypeError("Only dicts with str keys can be serialized")
            return {"__dict__": {k: encode(v) for k, v in value.items()}}
        if isinstance(value, tuple):
            return {"__tuple__": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        raise TypeError(f"Cannot serialize values of type {type(value).__name__}")

    header = json.dumps(encode(obj)).encode()
    blob = save_tensors(tensors) if tensors else b""
    return _MAGIC + _LENGTH.pack(len(header)) + header + blob


def loads(data: bytes, allowed: Iterable[type] = ()) -> Any:
    """Deserialize bytes produced by :func:`dumps`.

    Only dataclasses and enums listed in ``allowed`` can be rebuilt. Raises
    ``ValueError`` for anything malformed or not allowed.
    """
    registry = {_type_name(cls): cls for cls in allowed}
    start = len(_MAGIC) + _LENGTH.size
    if len(data) < start or not data.startswith(_MAGIC):
        raise ValueError("Not a LeRobot safe-serialization message")
    (header_len,) = _LENGTH.unpack_from(data, len(_MAGIC))
    if header_len > len(data) - start:
        raise ValueError("Truncated message")
    try:
        tree = json.loads(data[start : start + header_len])
        blob = data[start + header_len :]
        tensors = load_tensors(blob) if blob else {}
    except Exception as e:
        raise ValueError(f"Malformed message: {e}") from e

    def lookup(name: str, kind: str) -> type:
        cls = registry.get(name)
        if cls is None:
            raise ValueError(f"{kind} {name!r} is not allowed in this message")
        return cls

    def decode(node: Any) -> Any:
        if isinstance(node, list):
            return [decode(v) for v in node]
        if not isinstance(node, dict):
            return node
        if len(node) == 1:
            ((tag, payload),) = node.items()
            if tag == "__tensor__":
                return tensors[payload]
            if tag == "__ndarray__":
                return tensors[payload].numpy()
            if tag == "__bytes__":
                return base64.b64decode(payload, validate=True)
            if tag == "__tuple__":
                return tuple(decode(v) for v in payload)
            if tag == "__dict__":
                return {str(k): decode(v) for k, v in payload.items()}
        if node.keys() == {"__enum__", "value"}:
            cls = lookup(node["__enum__"], "Enum")
            if not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
                raise ValueError(f"{node['__enum__']!r} is not an enum")
            return cls(decode(node["value"]))
        if node.keys() == {"__dataclass__", "fields"}:
            cls = lookup(node["__dataclass__"], "Dataclass")
            if not dataclasses.is_dataclass(cls):
                raise ValueError(f"{node['__dataclass__']!r} is not a dataclass")
            return cls(**{str(k): decode(v) for k, v in node["fields"].items()})
        raise ValueError("Malformed message node")

    try:
        return decode(tree)
    except ValueError:
        raise
    except (KeyError, TypeError, AttributeError, RecursionError, base64.binascii.Error) as e:
        raise ValueError(f"Malformed message: {e}") from e
