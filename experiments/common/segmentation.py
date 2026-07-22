"""Classical OpenCV obstacle segmentation with optional manual correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy import ndimage

from .io_utils import resolve_path


SUPPORTED_SEGMENTATION_MODES = (
    "mask_file",
    "manual_polygon",
    "hsv",
    "background_reference",
)


@dataclass(frozen=True)
class SegmentationResult:
    """Raw and post-processed obstacle masks for one image."""

    raw_mask: np.ndarray
    clean_mask: np.ndarray
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = ensure_binary_mask(self.raw_mask)
        clean = ensure_binary_mask(self.clean_mask)
        if raw.shape != clean.shape:
            raise ValueError("Raw and cleaned masks must have identical shapes.")
        object.__setattr__(self, "raw_mask", raw)
        object.__setattr__(self, "clean_mask", clean)


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Return an 8-bit mask containing only 0 and 255."""

    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Binary masks must be two-dimensional, received shape {array.shape}.")
    if array.dtype == bool:
        return array.astype(np.uint8) * 255
    finite = np.nan_to_num(array.astype(float), nan=0.0, posinf=255.0, neginf=0.0)
    return (finite > 0.0).astype(np.uint8) * 255


def _odd_kernel_size(value: int | float | None) -> int:
    """Return a positive odd kernel size, or zero when filtering is disabled."""

    size = int(value or 0)
    if size <= 1:
        return 0
    return size if size % 2 == 1 else size + 1


def _remove_small_components(mask: np.ndarray, minimum_area_px: int) -> np.ndarray:
    """Remove connected foreground components smaller than a pixel-area threshold."""

    if minimum_area_px <= 0:
        return mask.copy()
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, count):
        area = int(statistics[label, cv2.CC_STAT_AREA])
        if area >= minimum_area_px:
            output[labels == label] = 255
    return output


def clean_binary_mask(raw_mask: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    """Apply deterministic blur, threshold, morphology, component, and hole filters."""

    cfg = config or {}
    mask = ensure_binary_mask(raw_mask)

    blur_size = _odd_kernel_size(cfg.get("blur_kernel", 0))
    if blur_size:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigmaX=0.0)

    threshold = int(cfg.get("threshold", 127))
    _value, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    open_size = _odd_kernel_size(cfg.get("open_kernel", 0))
    open_iterations = max(0, int(cfg.get("opening_iterations", cfg.get("open_iterations", 1))))
    if open_size and open_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    close_size = _odd_kernel_size(cfg.get("close_kernel", 0))
    close_iterations = max(0, int(cfg.get("closing_iterations", cfg.get("close_iterations", 1))))
    if close_size and close_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)

    mask = _remove_small_components(mask, int(cfg.get("minimum_component_area_px", 0)))

    if bool(cfg.get("fill_holes", False)):
        filled = ndimage.binary_fill_holes(mask > 0)
        mask = filled.astype(np.uint8) * 255

    if bool(cfg.get("invert", False)):
        mask = cv2.bitwise_not(mask)

    return ensure_binary_mask(mask)


def load_mask_file(path: str | Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a mask and resize it with nearest-neighbor interpolation."""

    mask_path = Path(path).expanduser().resolve()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file: {mask_path}")
    target_height, target_width = target_shape
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return ensure_binary_mask(mask)


def segment_hsv(image_bgr: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Threshold a BGR image in HSV color space."""

    lower = np.asarray(config.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.asarray(config.get("upper", [179, 255, 255]), dtype=np.uint8)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("HSV lower and upper bounds must each contain three values.")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    if bool(config.get("invert", False)):
        mask = cv2.bitwise_not(mask)
    return mask, {"lower": lower.tolist(), "upper": upper.tolist()}


def tune_hsv_interactive(
    image_bgr: np.ndarray,
    initial_config: dict[str, Any] | None = None,
    *,
    window_name: str = "HSV obstacle threshold",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune HSV bounds with OpenCV trackbars and return the accepted mask.

    The six trackbars directly control OpenCV HSV limits. Enter accepts the
    current mask, ``r`` restores the configured initial values, and Escape
    cancels the operation. The function is intentionally isolated so headless
    execution never creates a GUI object.
    """

    config = dict(initial_config or {})
    initial_lower = np.asarray(config.get("lower", [0, 0, 0]), dtype=np.uint8)
    initial_upper = np.asarray(config.get("upper", [179, 255, 255]), dtype=np.uint8)
    if initial_lower.shape != (3,) or initial_upper.shape != (3,):
        raise ValueError("HSV lower and upper bounds must each contain three values.")

    names = ("H low", "S low", "V low", "H high", "S high", "V high")
    maxima = (179, 255, 255, 179, 255, 255)
    initial_values = [
        int(initial_lower[0]), int(initial_lower[1]), int(initial_lower[2]),
        int(initial_upper[0]), int(initial_upper[1]), int(initial_upper[2]),
    ]
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        for name, maximum, value in zip(names, maxima, initial_values):
            cv2.createTrackbar(name, window_name, value, maximum, lambda _value: None)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        while True:
            values = [cv2.getTrackbarPos(name, window_name) for name in names]
            lower = np.asarray(values[:3], dtype=np.uint8)
            upper = np.asarray(values[3:], dtype=np.uint8)
            # Clamp crossed ranges rather than emitting an empty, confusing mask.
            upper = np.maximum(upper, lower)
            mask = cv2.inRange(hsv, lower, upper)
            if bool(config.get("invert", False)):
                mask = cv2.bitwise_not(mask)
            preview = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
            canvas = np.hstack([image_bgr, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), preview])
            cv2.putText(
                canvas,
                "Trackbars: HSV limits | Enter: accept | r: reset | Esc: cancel",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):
                return mask, {"lower": lower.tolist(), "upper": upper.tolist(), "interactive": True}
            if key == ord("r"):
                for name, value in zip(names, initial_values):
                    cv2.setTrackbarPos(name, window_name, value)
            elif key == 27:
                raise RuntimeError("Interactive HSV tuning was cancelled by the user.")
    finally:
        cv2.destroyWindow(window_name)


def segment_background_difference(
    image_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment foreground by absolute difference from a frozen background image."""

    if reference_bgr.shape[:2] != image_bgr.shape[:2]:
        reference_bgr = cv2.resize(
            reference_bgr,
            (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    use_color_norm = bool(config.get("use_color_norm", False))
    blur_size = _odd_kernel_size(config.get("difference_blur_kernel", 5))
    if use_color_norm:
        current = image_bgr
        reference = reference_bgr
        if blur_size:
            current = cv2.GaussianBlur(current, (blur_size, blur_size), 0.0)
            reference = cv2.GaussianBlur(reference, (blur_size, blur_size), 0.0)
        # Euclidean BGR change is more sensitive to chromatic foregrounds than
        # a grayscale difference. Values are clipped to an 8-bit diagnostic map.
        difference_float = np.linalg.norm(
            current.astype(np.float32) - reference.astype(np.float32),
            axis=2,
        )
        difference = np.clip(difference_float, 0.0, 255.0).astype(np.uint8)
    else:
        current_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        if blur_size:
            current_gray = cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0.0)
            reference_gray = cv2.GaussianBlur(reference_gray, (blur_size, blur_size), 0.0)
        difference = cv2.absdiff(current_gray, reference_gray)
    threshold = int(config.get("difference_threshold", 25))
    _value, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    return mask, {
        "difference_threshold": threshold,
        "difference_mean": float(np.mean(difference)),
        "difference_max": int(np.max(difference)),
        "use_color_norm": use_color_norm,
    }


def rasterize_polygons(image_shape: tuple[int, int], polygons: Iterable[Iterable[Iterable[float]]]) -> np.ndarray:
    """Rasterize one or more obstacle polygons into a binary mask."""

    mask = np.zeros(image_shape, dtype=np.uint8)
    contours: list[np.ndarray] = []
    for polygon in polygons:
        points = np.asarray(list(polygon), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
            raise ValueError("Each manual polygon must contain at least three two-dimensional points.")
        contours.append(np.round(points).astype(np.int32).reshape(-1, 1, 2))
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def select_polygons_interactive(image_bgr: np.ndarray, *, window_name: str = "Obstacle polygons") -> np.ndarray:
    """Draw obstacle polygons with the mouse and return a binary mask.

    Left click adds a vertex. ``n`` closes the current polygon. Enter completes
    all polygons. Backspace removes the most recent point. ``r`` clears every
    polygon, and Escape cancels.
    """

    committed: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def mouse_callback(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    try:
        while True:
            canvas = image_bgr.copy()
            for polygon in committed:
                contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(canvas, [contour], (0, 0, 255))
                cv2.polylines(canvas, [contour], True, (255, 255, 255), 2)
            if current:
                polyline = np.asarray(current, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [polyline], False, (0, 255, 255), 2)
                for point in current:
                    cv2.circle(canvas, point, 4, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                "Left click: vertex | n: close polygon | Enter: finish | Backspace: undo | r: reset",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("n") and len(current) >= 3:
                committed.append(current.copy())
                current.clear()
            elif key in (10, 13):
                if len(current) >= 3:
                    committed.append(current.copy())
                    current.clear()
                if committed:
                    return rasterize_polygons(image_bgr.shape[:2], committed)
            elif key in (8, 127) and current:
                current.pop()
            elif key == ord("r"):
                current.clear()
                committed.clear()
            elif key == 27:
                raise RuntimeError("Manual polygon segmentation was cancelled by the user.")
    finally:
        cv2.destroyWindow(window_name)


def edit_binary_mask(
    image_bgr: np.ndarray,
    initial_mask: np.ndarray,
    *,
    brush_radius_px: int = 12,
    window_name: str = "Manual mask correction",
) -> np.ndarray:
    """Interactively add or remove obstacle pixels with a circular brush.

    Left-drag paints occupied pixels. Right-drag erases occupied pixels. The
    bracket keys change brush size. Enter accepts, ``r`` restores the initial
    mask, and Escape cancels without modification.
    """

    original = ensure_binary_mask(initial_mask)
    mask = original.copy()
    radius = max(1, int(brush_radius_px))
    drawing = False
    erase = False

    state = {"radius": radius}

    def mouse_callback(event: int, x: int, y: int, flags: int, _parameter: object) -> None:
        nonlocal drawing, erase, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erase = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            erase = True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
        if drawing or (flags & cv2.EVENT_FLAG_LBUTTON) or (flags & cv2.EVENT_FLAG_RBUTTON):
            value = 0 if erase or (flags & cv2.EVENT_FLAG_RBUTTON) else 255
            cv2.circle(mask, (x, y), state["radius"], value, -1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    try:
        while True:
            overlay = image_bgr.copy()
            occupied = mask > 0
            red = np.zeros_like(overlay)
            red[..., 2] = 255
            overlay[occupied] = cv2.addWeighted(overlay[occupied], 0.35, red[occupied], 0.65, 0.0)
            cv2.putText(
                overlay,
                f"Left: add | Right: erase | [ ]: brush {state['radius']} px | Enter: accept | r: reset",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13):
                return ensure_binary_mask(mask)
            if key == ord("r"):
                mask = original.copy()
            elif key == ord("["):
                state["radius"] = max(1, state["radius"] - 2)
            elif key == ord("]"):
                state["radius"] += 2
            elif key == 27:
                return original
    finally:
        cv2.destroyWindow(window_name)


def load_background_reference(
    config: dict[str, Any],
    *,
    base_directory: str | Path,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Load and resize a configured background reference image."""

    path = resolve_path(config.get("reference_file"), base_directory=base_directory)
    if path is None:
        raise ValueError("background_reference mode requires segmentation.reference_file.")
    reference = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(f"Could not read background reference: {path}")
    if reference.shape[:2] != target_shape:
        reference = cv2.resize(reference, (target_shape[1], target_shape[0]), cv2.INTER_LINEAR)
    return reference


def segment_image(
    image_bgr: np.ndarray,
    config: dict[str, Any],
    *,
    base_directory: str | Path,
    background_reference: np.ndarray | None = None,
    allow_interactive: bool = True,
) -> SegmentationResult:
    """Run the configured segmentation method and common cleanup pipeline."""

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot segment an empty image.")
    mode = str(config.get("mode", "hsv"))
    if mode not in SUPPORTED_SEGMENTATION_MODES:
        raise ValueError(f"Unsupported segmentation mode {mode!r}; choose from {SUPPORTED_SEGMENTATION_MODES}.")

    metadata: dict[str, Any] = {"mode": mode}
    if mode == "mask_file":
        mask_path = resolve_path(config.get("mask_file"), base_directory=base_directory)
        if mask_path is None:
            raise ValueError("mask_file mode requires segmentation.mask_file.")
        raw = load_mask_file(mask_path, image_bgr.shape[:2])
        metadata["mask_file"] = str(mask_path)
    elif mode == "manual_polygon":
        configured_polygons = config.get("polygons")
        if configured_polygons:
            raw = rasterize_polygons(image_bgr.shape[:2], configured_polygons)
            metadata["polygon_count"] = len(configured_polygons)
        elif allow_interactive:
            raw = select_polygons_interactive(image_bgr)
        else:
            raise RuntimeError("manual_polygon mode requires configured polygons when running headlessly.")
    elif mode == "hsv":
        hsv_config = dict(config.get("hsv", {}))
        tune_interactively = bool(
            config.get("tune_hsv_interactively", hsv_config.get("tune_interactively", False))
        )
        if tune_interactively:
            if not allow_interactive:
                raise RuntimeError("Interactive HSV tuning cannot run in headless mode.")
            raw, details = tune_hsv_interactive(image_bgr, hsv_config)
        else:
            raw, details = segment_hsv(image_bgr, hsv_config)
        metadata.update(details)
    else:
        reference = background_reference
        if reference is None:
            reference = load_background_reference(
                config,
                base_directory=base_directory,
                target_shape=image_bgr.shape[:2],
            )
        raw, details = segment_background_difference(
            image_bgr,
            reference,
            config.get("background", {}),
        )
        metadata.update(details)

    clean = clean_binary_mask(raw, config.get("cleanup", {}))
    if bool(config.get("manual_correction", {}).get("enabled", False)):
        if not allow_interactive:
            raise RuntimeError("Manual mask correction cannot run in headless mode.")
        clean = edit_binary_mask(
            image_bgr,
            clean,
            brush_radius_px=int(config.get("manual_correction", {}).get("brush_radius_px", 12)),
        )

    metadata.update(
        {
            "raw_occupied_fraction": float(np.mean(raw > 0)),
            "clean_occupied_fraction": float(np.mean(clean > 0)),
        }
    )
    return SegmentationResult(raw_mask=raw, clean_mask=clean, mode=mode, metadata=metadata)
