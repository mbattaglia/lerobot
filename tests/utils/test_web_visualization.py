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
"""The local web viewer serves live data read-only, to loopback hosts only."""

import http.client
import json
import time

import cv2
import numpy as np
import pytest

from lerobot.utils import web_visualization as web
from lerobot.utils.visualization_utils import (
    init_visualization,
    log_visualization_data,
    shutdown_visualization,
)


@pytest.fixture
def viewer():
    url = web.init_web(port=0)
    port = int(url.rsplit(":", 1)[1])
    yield port
    web.shutdown_web()


def _get(port, path, host=None, method="GET"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host} if host else {}
    conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp, body


def _frame(h=48, w=64, color=(255, 0, 0)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


def test_page_and_assets_are_served_with_strict_headers(viewer):
    for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/app.css", "text/css")):
        resp, body = _get(viewer, path)
        assert resp.status == 200 and resp.getheader("Content-Type").startswith(kind) and body
        csp = resp.getheader("Content-Security-Policy")
        assert "default-src 'none'" in csp and "script-src 'self'" in csp
        assert resp.getheader("Access-Control-Allow-Origin") is None
        assert resp.getheader("X-Frame-Options") == "DENY"


def test_state_reports_latest_values(viewer):
    observation = {"shoulder_pan.pos": 12.5, "gripper.pos": 40.0, "front": _frame()}
    action = {"shoulder_pan.pos": 13.0, "gripper.pos": 41.0}
    web.log_web_data(observation=observation, action=action)
    state = json.loads(_get(viewer, "/state.json")[1])
    assert state["cameras"] == ["front"]
    assert state["observation"] == {"shoulder_pan.pos": 12.5, "gripper.pos": 40.0}
    assert state["action"] == {"shoulder_pan.pos": 13.0, "gripper.pos": 41.0}
    assert state["updates"] == 1


def test_camera_frame_is_a_jpeg_in_rgb_order(viewer):
    web.log_web_data(observation={"front": _frame(color=(255, 0, 0))})  # pure red, RGB
    resp, body = _get(viewer, "/camera/front.jpg")
    assert resp.status == 200 and resp.getheader("Content-Type") == "image/jpeg"
    bgr = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert bgr.shape == (48, 64, 3)
    b, g, r = (int(c) for c in bgr[24, 32])
    assert r > 200 and g < 50 and b < 50


def test_chw_and_single_channel_images_are_accepted(viewer):
    web.log_web_data(
        observation={"chw": np.zeros((3, 48, 64), np.uint8), "depth": np.zeros((48, 64, 1), np.uint16)}
    )
    assert _get(viewer, "/camera/chw.jpg")[0].status == 200
    assert _get(viewer, "/camera/depth.jpg")[0].status == 200


def test_stream_sends_multipart_jpeg_frames(viewer):
    web.log_web_data(observation={"front": _frame()})
    conn = http.client.HTTPConnection("127.0.0.1", viewer, timeout=5)
    conn.request("GET", "/stream/front")
    resp = conn.getresponse()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "multipart/x-mixed-replace; boundary=frame"
    chunk = resp.read(200)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    conn.close()


def test_favicon_request_is_answered_without_content(viewer):
    assert _get(viewer, "/favicon.ico")[0].status == 204


def test_unknown_paths_and_cameras_return_404(viewer):
    assert _get(viewer, "/camera/missing.jpg")[0].status == 404
    assert _get(viewer, "/stream/missing")[0].status == 404
    assert _get(viewer, "/../etc/passwd")[0].status == 404


@pytest.mark.parametrize("host", ["attacker.example:9876", "localhost.attacker.example", "192.168.1.10:9876"])
def test_foreign_host_header_is_rejected(viewer, host):
    assert _get(viewer, "/state.json", host=host)[0].status == 403


@pytest.mark.parametrize("host", ["127.0.0.1:9876", "localhost:9876", "localhost"])
def test_loopback_host_header_is_accepted(viewer, host):
    assert _get(viewer, "/state.json", host=host)[0].status == 200


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "OPTIONS"])
def test_viewer_is_read_only(viewer, method):
    assert _get(viewer, "/state.json", method=method)[0].status == 405


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_non_loopback_bind_is_refused(host):
    with pytest.raises(ValueError, match="loopback"):
        web.init_web(host=host, port=0)


def test_logging_without_a_viewer_is_a_no_op():
    web.shutdown_web()
    web.log_web_data(observation={"gripper.pos": 1.0})


def test_logging_is_cheap_for_the_control_loop(viewer):
    observation = {f"j{i}.pos": float(i) for i in range(6)} | {
        "front": _frame(480, 640),
        "wrist": _frame(480, 640),
    }
    start = time.perf_counter()
    for _ in range(200):
        web.log_web_data(observation=observation, action={"j0.pos": 1.0})
    assert (time.perf_counter() - start) / 200 < 0.002  # far below a 33 ms frame at 30 fps


def test_dispatch_selects_the_web_backend():
    init_visualization("web", ip="127.0.0.1", port=0)
    try:
        log_visualization_data("web", observation={"gripper.pos": 2.0}, action={"gripper.pos": 3.0})
        assert web._data.snapshot()["action"] == {"gripper.pos": 3.0}
    finally:
        shutdown_visualization("web")
    assert web._server is None
