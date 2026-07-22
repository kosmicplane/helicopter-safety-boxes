#!/usr/bin/env python3
"""
Patch vision_poisson_experiments so the live MJPEG/RTSP source is reopened
after the blocking calibration and mission-selection windows.

Why:
    The stream is opened before calibration. While the user clicks the workspace
    corners and landing zones, some backends buffer old MJPEG frames. The live
    loop then starts from the original first frame and may appear frozen.

What this patch does:
    1. Adds CAP_PROP_BUFFERSIZE=1 after cv2.VideoCapture(...), when possible.
    2. Reopens the source after mission setup.
    3. Reads a fresh frame before starting the live loop.
    4. Creates a timestamped backup before modifying the file.

Usage:
    cd ~/ATMOS/Docker/workspace/Helicopter/vision_poisson_experiments
    python fix_live_stream_refresh.py
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


TARGET = Path("common/live_pipeline.py")


def fail(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(1)


def add_capture_buffer_limit(text: str) -> tuple[str, bool]:
    """
    Add CAP_PROP_BUFFERSIZE=1 immediately after VideoCapture creation.

    Supports common forms:
        self.capture = cv2.VideoCapture(self.source)
        capture = cv2.VideoCapture(source)
    """
    if "CAP_PROP_BUFFERSIZE" in text:
        return text, False

    patterns = [
        (
            r"(?P<indent>[ \t]*)self\.capture\s*=\s*cv2\.VideoCapture\(self\.source\)\s*\n",
            (
                r"\g<indent>self.capture = cv2.VideoCapture(self.source)\n"
                r"\g<indent># Minimize buffered frames for MJPEG/RTSP sources.\n"
                r"\g<indent># Some OpenCV backends may ignore this property.\n"
                r"\g<indent>self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)\n"
            ),
        ),
        (
            r"(?P<indent>[ \t]*)capture\s*=\s*cv2\.VideoCapture\(source\)\s*\n",
            (
                r"\g<indent>capture = cv2.VideoCapture(source)\n"
                r"\g<indent># Minimize buffered frames for MJPEG/RTSP sources.\n"
                r"\g<indent># Some OpenCV backends may ignore this property.\n"
                r"\g<indent>capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)\n"
            ),
        ),
    ]

    for pattern, replacement in patterns:
        updated, count = re.subn(pattern, replacement, text, count=1)
        if count:
            return updated, True

    print(
        "[WARNING] Could not find the VideoCapture construction automatically. "
        "The stream-reopen fix will still be applied."
    )
    return text, False


def add_stream_reopen_after_setup(text: str) -> tuple[str, bool]:
    """
    Replace the stale first-frame assignment with a fresh source reopen.

    Expected original line:
        pending_frame: np.ndarray | None = first_frame

    The patch inserts source reopening before that line.
    """
    marker = "The video source could not provide a fresh frame after mission setup."
    if marker in text:
        return text, False

    pattern = (
        r"(?P<indent>[ \t]*)pending_frame\s*:\s*np\.ndarray\s*\|\s*None"
        r"\s*=\s*first_frame\s*\n"
    )

    match = re.search(pattern, text)
    if not match:
        fail(
            "Could not find `pending_frame: np.ndarray | None = first_frame` in "
            f"{TARGET}. The file structure differs from the expected version."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}# Reopen the source after the blocking calibration/mission UI.\n"
        f"{indent}# This discards buffered MJPEG/RTSP frames accumulated while\n"
        f"{indent}# the user was selecting corners, START, and landing zones.\n"
        f"{indent}video.release()\n"
        f"{indent}video = VideoSource(\n"
        f"{indent}    self.source_value,\n"
        f"{indent}    reconnection_attempts=int(source_cfg.get(\"reconnection_attempts\", 2)),\n"
        f"{indent}    reconnection_delay_s=float(source_cfg.get(\"reconnection_delay_s\", 0.2)),\n"
        f"{indent})\n"
        f"{indent}\n"
        f"{indent}ok, fresh_frame, _timestamp = video.read()\n"
        f"{indent}if not ok or fresh_frame is None:\n"
        f"{indent}    raise RuntimeError(\n"
        f"{indent}        \"The video source could not provide a fresh frame after mission setup.\"\n"
        f"{indent}    )\n"
        f"{indent}\n"
        f"{indent}pending_frame: np.ndarray | None = fresh_frame\n"
    )

    updated, count = re.subn(pattern, replacement, text, count=1)
    return updated, bool(count)


def main() -> None:
    if not TARGET.exists():
        fail(
            f"{TARGET} was not found. Run this script from the root of "
            "vision_poisson_experiments."
        )

    original = TARGET.read_text(encoding="utf-8")
    updated = original

    updated, buffer_changed = add_capture_buffer_limit(updated)
    updated, reopen_changed = add_stream_reopen_after_setup(updated)

    if updated == original:
        print("[INFO] No changes were necessary; the fix appears to be installed already.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"{TARGET.name}.backup_{timestamp}")
    shutil.copy2(TARGET, backup)

    TARGET.write_text(updated, encoding="utf-8")

    print("[OK] Live-stream refresh patch applied.")
    print(f"[OK] Backup created: {backup}")
    print(f"[INFO] CAP_PROP_BUFFERSIZE change: {'applied' if buffer_changed else 'already present/not found'}")
    print(f"[INFO] Post-setup source reopen: {'applied' if reopen_changed else 'already present'}")
    print()
    print("Run the live experiment with:")
    print(
        'python 02_phone_stream_poisson_realtime/run_experiment.py '
        '--source "http://192.168.1.220:8080/stream.mjpg" '
        '--config 02_phone_stream_poisson_realtime/config.yaml --verbose'
    )


if __name__ == "__main__":
    main()
