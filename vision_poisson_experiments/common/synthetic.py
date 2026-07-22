"""Deterministic synthetic images and videos used by tests and demonstrations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _draw_workspace_background(size_px: tuple[int, int]) -> np.ndarray:
    """Create a textured but deterministic top-down workspace image."""

    width, height = size_px
    image = np.full((height, width, 3), 232, dtype=np.uint8)
    for x in range(0, width, 40):
        cv2.line(image, (x, 0), (x, height - 1), (210, 210, 210), 1)
    for y in range(0, height, 40):
        cv2.line(image, (0, y), (width - 1, y), (210, 210, 210), 1)
    cv2.rectangle(image, (5, 5), (width - 6, height - 6), (80, 80, 80), 3)
    cv2.putText(
        image,
        "Synthetic fixed top-down workspace",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )
    return image


def generate_static_scene(output_directory: str | Path, *, size_px: tuple[int, int] = (640, 480)) -> dict[str, Path]:
    """Generate a static BGR scene, exact obstacle mask, and empty reference."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    width, height = size_px
    background = _draw_workspace_background(size_px)
    scene = background.copy()
    mask = np.zeros((height, width), dtype=np.uint8)

    # A tall central obstacle makes the straight start-to-goal path unsafe while
    # leaving free corridors above and below it.
    cv2.rectangle(scene, (285, 90), (350, 350), (35, 85, 180), -1)
    cv2.rectangle(mask, (285, 90), (350, 350), 255, -1)

    cv2.circle(scene, (455, 145), 46, (45, 145, 70), -1)
    cv2.circle(mask, (455, 145), 46, 255, -1)

    polygon = np.asarray([[105, 305], [190, 275], [215, 380], [130, 400]], dtype=np.int32)
    cv2.fillPoly(scene, [polygon], (145, 70, 45))
    cv2.fillPoly(mask, [polygon], 255)

    cv2.imwrite(str(output / "static_background.png"), background)
    cv2.imwrite(str(output / "static_scene.png"), scene)
    cv2.imwrite(str(output / "static_mask.png"), mask)
    return {
        "background": output / "static_background.png",
        "scene": output / "static_scene.png",
        "mask": output / "static_mask.png",
    }


def generate_live_video(
    output_directory: str | Path,
    *,
    size_px: tuple[int, int] = (640, 480),
    frame_count: int = 120,
    fps: float = 20.0,
) -> dict[str, Path]:
    """Generate a fixed-camera video with deterministic moving obstacles."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    width, height = size_px
    background = _draw_workspace_background(size_px)
    background_path = output / "live_background.png"
    video_path = output / "live_scene.avi"
    cv2.imwrite(str(background_path), background)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video at {video_path}")
    try:
        for frame_index in range(int(frame_count)):
            frame = background.copy()
            phase = 2.0 * np.pi * frame_index / max(1, frame_count - 1)
            center_x = int(round(160 + 260 * (0.5 + 0.5 * np.sin(phase))))
            center_y = int(round(235 + 95 * np.cos(phase)))
            cv2.circle(frame, (center_x, center_y), 36, (20, 40, 210), -1)

            rectangle_x = int(round(430 + 70 * np.sin(1.7 * phase)))
            rectangle_y = int(round(120 + 45 * np.cos(1.3 * phase)))
            cv2.rectangle(
                frame,
                (rectangle_x - 32, rectangle_y - 45),
                (rectangle_x + 32, rectangle_y + 45),
                (40, 155, 55),
                -1,
            )
            cv2.putText(
                frame,
                f"frame {frame_index:03d}",
                (width - 150, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (50, 50, 50),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    return {"background": background_path, "video": video_path}


def generate_all_assets(output_directory: str | Path) -> dict[str, Path]:
    """Generate every synthetic asset and return a flat path mapping."""

    static = generate_static_scene(output_directory)
    live = generate_live_video(output_directory)
    return {**{f"static_{key}": value for key, value in static.items()}, **{f"live_{key}": value for key, value in live.items()}}
