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
"""Local web viewer: camera feeds and joint values in a browser, with no extra packages.

Select it with ``--display_data=true --display_mode=web`` and open the printed URL (default
``http://127.0.0.1:9876``). It uses only the standard library and OpenCV, which LeRobot already needs.

The server is local and read-only:
- it binds only to a loopback address and refuses any other bind address;
- it answers only requests whose ``Host`` is a loopback name, which blocks DNS rebinding;
- it serves only ``GET``, sends no CORS headers, and sets a strict Content-Security-Policy.

The control loop only stores references to the latest values (:func:`log_web_data`).
JPEG encoding runs in the server's threads, at most once per new frame and camera.
"""

import json
import logging
import numbers
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
LOOPBACK_BIND = frozenset({"127.0.0.1", "localhost", "::1"})
LOOPBACK_HOST_NAMES = frozenset({"127.0.0.1", "localhost", "[::1]"})
STREAM_MAX_FPS = 15
JPEG_QUALITY = 80

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeRobot live view</title>
<link rel="stylesheet" href="/app.css">
</head><body>
<header><h1>LeRobot live view</h1><span id="status">connecting</span></header>
<main><section id="cameras"></section><section><table id="values">
<thead><tr><th>name</th><th>observation</th><th>action</th></tr></thead><tbody></tbody></table></section></main>
<script src="/app.js"></script>
</body></html>
"""

_CSS = """
:root { color-scheme: light dark; --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --card: #ffffff; --line: #deded8; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ececea; --muted: #a3a39d; --card: #222220; --line: #3a3a37; } }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
header { display: flex; gap: 12px; align-items: baseline; padding: 12px 16px; border-bottom: 1px solid var(--line); }
h1 { font-size: 16px; margin: 0; }
#status { color: var(--muted); }
main { display: grid; grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr); gap: 16px; padding: 16px; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
#cameras { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
figure { margin: 0; background: var(--card); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
figure img { display: block; width: 100%; height: auto; background: #000; }
figcaption { padding: 6px 10px; color: var(--muted); }
table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--line); text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; }
"""

_JS = """
const cams = document.getElementById("cameras");
const body = document.querySelector("#values tbody");
const status = document.getElementById("status");
let shown = "";
const fmt = (v) => (v === undefined || v === null ? "" : Number(v).toFixed(2));
async function tick() {
  try {
    const r = await fetch("/state.json", { cache: "no-store" });
    const s = await r.json();
    const names = s.cameras.join("|");
    if (names !== shown) {
      shown = names;
      cams.replaceChildren(...s.cameras.map((c) => {
        const f = document.createElement("figure");
        const img = document.createElement("img");
        img.src = "/stream/" + encodeURIComponent(c);
        img.alt = c;
        const cap = document.createElement("figcaption");
        cap.textContent = c;
        f.append(img, cap);
        return f;
      }));
    }
    const keys = [...new Set([...Object.keys(s.observation), ...Object.keys(s.action)])].sort();
    body.replaceChildren(...keys.map((k) => {
      const tr = document.createElement("tr");
      for (const v of [k, fmt(s.observation[k]), fmt(s.action[k])]) {
        const td = document.createElement("td");
        td.textContent = v;
        tr.append(td);
      }
      return tr;
    }));
    status.textContent = s.updates ? `${s.updates} updates, last ${s.age_s.toFixed(1)} s ago` : "waiting for data";
  } catch (e) {
    status.textContent = "not connected";
  }
  setTimeout(tick, 200);
}
tick();
"""


class _LatestData:
    """The newest observation/action values, shared between the control loop and the server."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.images: dict[str, np.ndarray] = {}
        self.versions: dict[str, int] = {}
        self.observation: dict[str, float] = {}
        self.action: dict[str, float] = {}
        self.updates = 0
        self.last_update = 0.0
        self._jpeg_cache: dict[str, tuple[int, bytes]] = {}

    def update(self, images: dict[str, np.ndarray], observation: dict[str, float], action: dict[str, float]):
        with self.cond:
            for name, img in images.items():
                self.images[name] = img
                self.versions[name] = self.versions.get(name, 0) + 1
            self.observation.update(observation)
            self.action.update(action)
            self.updates += 1
            self.last_update = time.monotonic()
            self.cond.notify_all()

    def snapshot(self) -> dict:
        with self.cond:
            return {
                "cameras": sorted(self.images),
                "observation": dict(self.observation),
                "action": dict(self.action),
                "updates": self.updates,
                "age_s": time.monotonic() - self.last_update if self.updates else 0.0,
            }

    def jpeg(self, name: str) -> tuple[int, bytes] | None:
        with self.cond:
            img, version = self.images.get(name), self.versions.get(name, 0)
            cached = self._jpeg_cache.get(name)
        if img is None:
            return None
        if cached is not None and cached[0] == version:
            return cached
        ok, buf = cv2.imencode(".jpg", _to_bgr(img), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        encoded = (version, buf.tobytes())
        with self.cond:
            self._jpeg_cache[name] = encoded
        return encoded

    def wait_for_new(self, name: str, version: int, timeout: float) -> int:
        with self.cond:
            self.cond.wait_for(lambda: self.versions.get(name, 0) != version, timeout=timeout)
            return self.versions.get(name, 0)


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """RGB (LeRobot's camera format) or single-channel image -> BGR uint8 for OpenCV."""
    if img.dtype != np.uint8:
        lo, hi = float(np.min(img)), float(np.max(img))
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros(img.shape, np.uint8)
    if img.ndim == 2 or img.shape[-1] == 1:
        return cv2.cvtColor(img.reshape(img.shape[0], img.shape[1]), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2BGR)


def _host_is_loopback(host_header: str | None) -> bool:
    if not host_header:
        return False
    host = host_header.strip().lower()
    host = host.split("]", 1)[0] + "]" if host.startswith("[") else host.split(":", 1)[0]
    return host in LOOPBACK_HOST_NAMES


class _Handler(BaseHTTPRequestHandler):
    server_version = "LeRobotWebView"
    sys_version = ""
    data: _LatestData  # set on the subclass created in init_web

    def log_message(self, format, *args):  # noqa: A002 - matches the base-class signature
        logger.debug("web viewer: " + format, *args)

    def _headers(self, status: int, content_type: str, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        )
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self._headers(status, content_type, len(body))
        self.wfile.write(body)

    def _reject_non_get(self) -> None:
        self._send(405, "text/plain; charset=utf-8", b"read-only viewer\n")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _reject_non_get  # noqa: N815 - http.server names

    def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        if not _host_is_loopback(self.headers.get("Host")):
            self._send(403, "text/plain; charset=utf-8", b"host not allowed\n")
            return
        path = urlsplit(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", _PAGE.encode())
        elif path == "/app.js":
            self._send(200, "text/javascript; charset=utf-8", _JS.encode())
        elif path == "/app.css":
            self._send(200, "text/css; charset=utf-8", _CSS.encode())
        elif path == "/favicon.ico":
            self._send(204, "image/x-icon", b"")
        elif path == "/state.json":
            self._send(200, "application/json", json.dumps(self.data.snapshot()).encode())
        elif path.startswith("/camera/") and path.endswith(".jpg"):
            frame = self.data.jpeg(unquote(path[len("/camera/") : -len(".jpg")]))
            if frame is None:
                self._send(404, "text/plain; charset=utf-8", b"no such camera\n")
            else:
                self._send(200, "image/jpeg", frame[1])
        elif path.startswith("/stream/"):
            self._stream(unquote(path[len("/stream/") :]))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def _stream(self, name: str) -> None:
        if self.data.jpeg(name) is None:
            self._send(404, "text/plain; charset=utf-8", b"no such camera\n")
            return
        self._headers(200, "multipart/x-mixed-replace; boundary=frame")
        version = -1
        try:
            while not self.server.stopping.is_set():
                version = self.data.wait_for_new(name, version, timeout=1.0)
                frame = self.data.jpeg(name)
                if frame is None:
                    break
                version = frame[0]
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame[1])).encode()
                    + b"\r\n\r\n"
                    + frame[1]
                    + b"\r\n"
                )
                self.wfile.flush()
                time.sleep(1.0 / STREAM_MAX_FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler):
        if ":" in address[0]:
            import socket

            self.address_family = socket.AF_INET6
        super().__init__(address, handler)
        self.stopping = threading.Event()


_server: _Server | None = None
_data: _LatestData | None = None


def init_web(host: str = DEFAULT_HOST, port: int | None = DEFAULT_PORT) -> str:
    """Start the local viewer and return its URL. Refuses non-loopback bind addresses."""
    global _server, _data
    if host not in LOOPBACK_BIND:
        raise ValueError(
            f"The web viewer only binds to a loopback address ({sorted(LOOPBACK_BIND)}), not {host!r}. "
            "Use an SSH tunnel to view it from another machine."
        )
    if _server is not None:
        shutdown_web()
    _data = _LatestData()
    handler = type("_BoundHandler", (_Handler,), {"data": _data})
    _server = _Server((host, DEFAULT_PORT if port is None else port), handler)
    threading.Thread(target=_server.serve_forever, name="lerobot-web-viewer", daemon=True).start()
    bound_port = _server.server_address[1]
    url = f"http://{'[::1]' if ':' in host else host}:{bound_port}"
    print(f"Live view: {url}")
    return url


def _split(values: dict | None, prefix: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    images, scalars = {}, {}
    for key, value in (values or {}).items():
        if value is None:
            continue
        name = str(key).removeprefix(prefix)
        if isinstance(value, numbers.Real) or (isinstance(value, np.ndarray) and value.ndim == 0):
            scalars[name] = float(value)
        elif isinstance(value, np.ndarray):
            arr = value
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 1:
                for i, v in enumerate(arr.tolist()):
                    scalars[f"{name}[{i}]"] = float(v)
            elif arr.ndim in (2, 3):
                images[name] = arr
    return images, scalars


def log_web_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """Store the latest observation and action for the viewer. Cheap: no encoding happens here."""
    del compress_images  # frames are always JPEG-encoded for the browser
    if _data is None:
        return
    images, obs_scalars = _split(observation, "observation.")
    action_images, action_scalars = _split(action, "action.")
    images.update(action_images)
    _data.update(images, obs_scalars, action_scalars)


def shutdown_web() -> None:
    """Stop the viewer if it is running."""
    global _server, _data
    if _server is None:
        return
    _server.stopping.set()
    with _data.cond:
        _data.cond.notify_all()
    _server.shutdown()
    _server.server_close()
    _server, _data = None, None
