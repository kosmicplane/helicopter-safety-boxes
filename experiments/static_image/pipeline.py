"""Static-image perception and safety-field construction.

The pipeline converts one calibrated image into a metric obstacle mask,
configuration-space occupancy, and a two-dimensional Poisson safety field.
It contains no controller logic; the reusable safety boxes consume its outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from safety_box_core import EquilibriumTarget
from experiments.common.calibration import assume_top_down_calibration, rectify_image
from experiments.common.coordinates import GridGeometry
from experiments.common.occupancy import OccupancyMaps, build_occupancy_maps
from experiments.common.poisson_field import PoissonField, compute_poisson_field
from experiments.common.segmentation import SegmentationResult, segment_image


@dataclass(frozen=True, slots=True)
class StaticImageProducts:
    """Self-contained outputs of the static-image perception stage."""

    source_image_bgr: np.ndarray
    rectified_image_bgr: np.ndarray
    segmentation: SegmentationResult
    occupancy: OccupancyMaps
    geometry: GridGeometry
    poisson_field: PoissonField
    targets: tuple[EquilibriumTarget, ...]


@dataclass(frozen=True, slots=True)
class PlanarWorld:
    """Minimal metric world contract required by shared plotting utilities."""

    extent_m: tuple[float, float]
    targets: tuple[EquilibriumTarget, ...]
    obstacles: tuple[Any, ...] = ()


def load_static_image(path: str | Path) -> np.ndarray:
    """Load a BGR image and reject missing or empty files."""

    image_path = Path(path).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not read input image: {image_path}")
    return image


def build_targets(config: Mapping[str, Any]) -> tuple[EquilibriumTarget, ...]:
    """Create zero-velocity landing equilibria from configured XY centers."""

    targets: list[EquilibriumTarget] = []
    for index, point in enumerate(config["landing_zones"]):
        xy = np.asarray(point, dtype=float).reshape(2)
        targets.append(
            EquilibriumTarget(
                identifier=f"LZ{index}",
                x_star=np.concatenate([xy, np.zeros(2)]),
                u_star=np.zeros(2),
                metadata={
                    "position_xy_m": xy.tolist(),
                    "landing_radius_m": float(config["landing_radius_m"]),
                },
            )
        )
    return tuple(targets)


def build_static_image_products(
    *,
    image_path: str | Path,
    experiment_config: Mapping[str, Any],
    poisson_config: Mapping[str, Any],
    repository_root: str | Path,
    forcing_method: str | None = None,
    solver: str | None = None,
) -> StaticImageProducts:
    """Run calibration, segmentation, inflation, and Poisson synthesis."""

    source = load_static_image(image_path)
    workspace = tuple(float(value) for value in experiment_config["workspace_size_m"])
    output_size = tuple(int(value) for value in experiment_config["output_size_px"])
    calibration = assume_top_down_calibration(
        source.shape,
        output_size_px=output_size,
        workspace_size_m=workspace,
    )
    rectified = rectify_image(source, calibration)
    segmentation = segment_image(
        rectified,
        dict(experiment_config["segmentation"]),
        base_directory=repository_root,
        allow_interactive=False,
    )

    # The image grid is stored in NumPy row-major (y,x) order.  The Poisson
    # controller interface uses physical coordinate order (x,y), so the
    # occupancy is transposed once at the package boundary and never again.
    grid_shape_yx = (output_size[1], output_size[0])
    occupancy_maps = build_occupancy_maps(
        segmentation.clean_mask,
        grid_shape_yx=grid_shape_yx,
        workspace_size_m=workspace,
        robot_radius_m=float(experiment_config["robot_radius_m"]),
        perception_margin_m=float(experiment_config["perception_margin_m"]),
    )
    geometry = GridGeometry(
        width_m=workspace[0],
        height_m=workspace[1],
        nx=grid_shape_yx[1],
        ny=grid_shape_yx[0],
    )
    occupancy_xy = occupancy_maps.inflated_occupancy.T
    spacing_xy = (geometry.dx, geometry.dy)
    field = compute_poisson_field(
        occupancy_xy,
        spacing=spacing_xy,
        config=poisson_config,
        forcing_method=forcing_method,
        solver=solver,
    )
    return StaticImageProducts(
        source_image_bgr=source,
        rectified_image_bgr=rectified,
        segmentation=segmentation,
        occupancy=occupancy_maps,
        geometry=geometry,
        poisson_field=field,
        targets=build_targets(experiment_config),
    )


def point_is_occupied(products: StaticImageProducts, point_xy: np.ndarray) -> bool:
    """Return whether a physical XY point lies outside the map or in occupancy."""

    point = np.asarray(point_xy, dtype=float).reshape(2)
    if not products.geometry.contains_xy(point):
        return True
    row, column = products.geometry.nearest_index_yx(point, clip=True)
    return bool(products.occupancy.inflated_occupancy[row, column])


def save_perception_products(products: StaticImageProducts, directory: str | Path) -> None:
    """Persist image, masks, occupancy, and calibration-independent metadata."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "source_image.png"), products.source_image_bgr)
    cv2.imwrite(str(output / "rectified_image.png"), products.rectified_image_bgr)
    cv2.imwrite(str(output / "raw_mask.png"), products.segmentation.raw_mask)
    cv2.imwrite(str(output / "clean_mask.png"), products.segmentation.clean_mask)
    cv2.imwrite(
        str(output / "inflated_occupancy.png"),
        products.occupancy.inflated_occupancy.astype(np.uint8) * 255,
    )
    np.savez_compressed(
        output / "occupancy_data.npz",
        occupancy=products.occupancy.occupancy,
        inflated_occupancy=products.occupancy.inflated_occupancy,
        spacing_yx=np.asarray(products.occupancy.grid_spacing_yx),
    )
    products.poisson_field.save(output)
