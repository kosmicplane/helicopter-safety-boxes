"""Top-down workspace calibration and global camera-motion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


_POINT_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")


@dataclass(frozen=True)
class CalibrationData:
    """Serializable perspective calibration for a fixed planar workspace."""

    source_points_px: np.ndarray
    destination_points_px: np.ndarray
    homography: np.ndarray
    output_size_px: tuple[int, int]
    workspace_size_m: tuple[float, float]
    mode: str

    def __post_init__(self) -> None:
        source = np.asarray(self.source_points_px, dtype=np.float64)
        destination = np.asarray(self.destination_points_px, dtype=np.float64)
        homography = np.asarray(self.homography, dtype=np.float64)
        if source.shape != (4, 2) or destination.shape != (4, 2):
            raise ValueError("Calibration points must have shape (4, 2).")
        if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
            raise ValueError("Calibration homography must be a finite 3x3 matrix.")
        width_px, height_px = self.output_size_px
        width_m, height_m = self.workspace_size_m
        if width_px < 2 or height_px < 2:
            raise ValueError("Rectified output dimensions must be at least two pixels.")
        if width_m <= 0.0 or height_m <= 0.0:
            raise ValueError("Workspace dimensions must be positive.")
        object.__setattr__(self, "source_points_px", source)
        object.__setattr__(self, "destination_points_px", destination)
        object.__setattr__(self, "homography", homography)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation."""

        return {
            "source_points_px": self.source_points_px.tolist(),
            "destination_points_px": self.destination_points_px.tolist(),
            "homography": self.homography.tolist(),
            "output_size_px": list(self.output_size_px),
            "workspace_size_m": list(self.workspace_size_m),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationData":
        """Construct a calibration from serialized data."""

        return cls(
            source_points_px=np.asarray(data["source_points_px"], dtype=float),
            destination_points_px=np.asarray(data["destination_points_px"], dtype=float),
            homography=np.asarray(data["homography"], dtype=float),
            output_size_px=(int(data["output_size_px"][0]), int(data["output_size_px"][1])),
            workspace_size_m=(
                float(data["workspace_size_m"][0]),
                float(data["workspace_size_m"][1]),
            ),
            mode=str(data.get("mode", "load_file")),
        )

    def save(self, path: str | Path) -> None:
        """Save the calibration as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationData":
        """Load and validate a calibration JSON file."""

        input_path = Path(path).expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Calibration file does not exist: {input_path}")
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Calibration JSON must contain an object.")
        return cls.from_dict(data)


def destination_corners(output_size_px: tuple[int, int]) -> np.ndarray:
    """Return rectified image corners in the required clockwise order."""

    width_px, height_px = output_size_px
    return np.asarray(
        [
            [0.0, 0.0],
            [float(width_px - 1), 0.0],
            [float(width_px - 1), float(height_px - 1)],
            [0.0, float(height_px - 1)],
        ],
        dtype=np.float32,
    )


def validate_quadrilateral(points: Iterable[Iterable[float]], *, minimum_area_px2: float = 100.0) -> np.ndarray:
    """Validate point order, finite values, convexity, and non-degenerate area."""

    array = np.asarray(list(points), dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"Expected exactly four 2D points, received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Calibration points contain NaN or infinity.")
    contour = array.reshape(-1, 1, 2)
    area = abs(float(cv2.contourArea(contour)))
    if area < minimum_area_px2:
        raise ValueError(
            f"Calibration quadrilateral is degenerate: area {area:.3f} px^2 is below "
            f"the minimum {minimum_area_px2:.3f} px^2."
        )
    if not cv2.isContourConvex(contour.astype(np.int32)):
        raise ValueError(
            "Calibration points do not form a convex quadrilateral in the required order: "
            "top-left, top-right, bottom-right, bottom-left."
        )
    edge_lengths = np.linalg.norm(np.roll(array, -1, axis=0) - array, axis=1)
    if np.min(edge_lengths) < 2.0:
        raise ValueError("Two calibration points are too close to each other.")
    return array


def build_calibration(
    source_points_px: Iterable[Iterable[float]],
    *,
    output_size_px: tuple[int, int],
    workspace_size_m: tuple[float, float],
    mode: str,
) -> CalibrationData:
    """Construct a perspective calibration from four ordered image points."""

    source = validate_quadrilateral(source_points_px)
    destination = destination_corners(output_size_px)
    homography = cv2.getPerspectiveTransform(source, destination)
    if not np.all(np.isfinite(homography)) or abs(float(np.linalg.det(homography))) < 1.0e-12:
        raise ValueError("Computed homography is singular or numerically invalid.")
    return CalibrationData(
        source_points_px=source,
        destination_points_px=destination,
        homography=homography,
        output_size_px=output_size_px,
        workspace_size_m=workspace_size_m,
        mode=mode,
    )


def assume_top_down_calibration(
    image_shape: tuple[int, ...],
    *,
    output_size_px: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> CalibrationData:
    """Map the entire image rectangle to the configured metric workspace."""

    if len(image_shape) < 2:
        raise ValueError("Image shape must contain height and width.")
    height_px, width_px = int(image_shape[0]), int(image_shape[1])
    source = np.asarray(
        [
            [0.0, 0.0],
            [float(width_px - 1), 0.0],
            [float(width_px - 1), float(height_px - 1)],
            [0.0, float(height_px - 1)],
        ],
        dtype=np.float32,
    )
    return build_calibration(
        source,
        output_size_px=output_size_px,
        workspace_size_m=workspace_size_m,
        mode="assume_top_down",
    )


def rectify_image(image: np.ndarray, calibration: CalibrationData, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Apply the stored homography to an image or binary mask."""

    if image is None or image.size == 0:
        raise ValueError("Cannot rectify an empty image.")
    return cv2.warpPerspective(
        image,
        calibration.homography,
        calibration.output_size_px,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def draw_calibration_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Annotate selected calibration points and their required order."""

    canvas = image.copy()
    for index, point in enumerate(np.asarray(points, dtype=float)):
        center = tuple(int(round(value)) for value in point)
        cv2.circle(canvas, center, 7, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{index + 1}: {_POINT_LABELS[index]}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) == 4:
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [contour], True, (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


def select_four_points_interactive(image: np.ndarray, *, window_name: str = "Workspace calibration") -> np.ndarray:
    """Collect four ordered points with an OpenCV mouse interface.

    Left click adds a point. ``r`` resets, Backspace removes the last point,
    Enter confirms four points, and Escape cancels.
    """

    points: list[tuple[float, float]] = []

    def mouse_callback(event: int, x: int, y: int, _flags: int, _parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    try:
        while True:
            canvas = draw_calibration_points(image, np.asarray(points, dtype=float)) if points else image.copy()
            instruction = (
                f"Click {_POINT_LABELS[len(points)]}" if len(points) < 4 else "Press Enter to accept"
            )
            cv2.putText(
                canvas,
                instruction + " | r: reset | Backspace: undo | Esc: cancel",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(points) == 4:
                return validate_quadrilateral(points)
            if key in (8, 127) and points:
                points.pop()
            elif key == ord("r"):
                points.clear()
            elif key == 27:
                raise RuntimeError("Interactive calibration was cancelled by the user.")
    finally:
        cv2.destroyWindow(window_name)


def interactive_calibration(
    image: np.ndarray,
    *,
    output_size_px: tuple[int, int],
    workspace_size_m: tuple[float, float],
) -> CalibrationData:
    """Run point selection and construct a validated perspective calibration."""

    points = select_four_points_interactive(image)
    return build_calibration(
        points,
        output_size_px=output_size_px,
        workspace_size_m=workspace_size_m,
        mode="interactive",
    )


@dataclass(frozen=True)
class MotionEstimate:
    """Result of a global affine-motion estimate between two rectified frames."""

    valid: bool
    moved: bool
    translation_px: float
    rotation_deg: float
    inlier_ratio: float
    match_count: int
    message: str


class GlobalMotionDetector:
    """Detect camera movement using ORB features and robust affine estimation.

    The detector is intentionally diagnostic rather than a stabilizer. A detected
    camera motion invalidates the metric interpretation of the current frame until
    the user recalibrates or deliberately accepts a new reference.
    """

    def __init__(
        self,
        reference_bgr: np.ndarray,
        *,
        translation_threshold_px: float = 4.0,
        rotation_threshold_deg: float = 1.5,
        minimum_matches: int = 20,
        maximum_features: int = 1000,
    ) -> None:
        self.translation_threshold_px = float(translation_threshold_px)
        self.rotation_threshold_deg = float(rotation_threshold_deg)
        self.minimum_matches = int(minimum_matches)
        self.orb = cv2.ORB_create(nfeatures=int(maximum_features))
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.set_reference(reference_bgr)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        """Convert a BGR or grayscale image into an 8-bit grayscale image."""

        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.asarray(gray, dtype=np.uint8)

    def set_reference(self, reference_bgr: np.ndarray) -> None:
        """Replace the fixed reference and precompute its ORB descriptors."""

        self.reference_gray = self._gray(reference_bgr)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(
            self.reference_gray,
            None,
        )

    def estimate(self, current_bgr: np.ndarray) -> MotionEstimate:
        """Estimate camera translation and rotation relative to the reference."""

        current_gray = self._gray(current_bgr)
        current_keypoints, current_descriptors = self.orb.detectAndCompute(current_gray, None)
        if self.reference_descriptors is None or current_descriptors is None:
            return MotionEstimate(False, False, 0.0, 0.0, 0.0, 0, "insufficient_descriptors")
        matches = self.matcher.match(self.reference_descriptors, current_descriptors)
        matches = sorted(matches, key=lambda match: match.distance)
        if len(matches) < self.minimum_matches:
            return MotionEstimate(
                False,
                False,
                0.0,
                0.0,
                0.0,
                len(matches),
                "insufficient_matches",
            )
        # Retaining the strongest matches makes the affine estimate less sensitive
        # to moving foreground obstacles that are visible in only part of the image.
        retained = matches[: min(len(matches), 300)]
        source = np.float32([self.reference_keypoints[m.queryIdx].pt for m in retained]).reshape(-1, 1, 2)
        destination = np.float32([current_keypoints[m.trainIdx].pt for m in retained]).reshape(-1, 1, 2)
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if affine is None:
            return MotionEstimate(False, False, 0.0, 0.0, 0.0, len(retained), "affine_estimation_failed")
        translation = float(np.hypot(affine[0, 2], affine[1, 2]))
        rotation = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
        inlier_ratio = float(np.mean(inliers)) if inliers is not None and inliers.size else 0.0
        moved = (
            translation > self.translation_threshold_px
            or abs(rotation) > self.rotation_threshold_deg
        )
        return MotionEstimate(
            True,
            moved,
            translation,
            rotation,
            inlier_ratio,
            len(retained),
            "camera_moved" if moved else "camera_stable",
        )
