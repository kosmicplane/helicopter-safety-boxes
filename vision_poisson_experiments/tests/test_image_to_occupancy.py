"""Synthetic segmentation-to-grid alignment and physical inflation tests."""

from __future__ import annotations

import cv2
import numpy as np

from common.coordinates import GridGeometry
from common.occupancy import compute_occupancy_products, inflate_occupancy, mask_to_occupancy


def test_rectangle_and_circle_align_after_nearest_resize() -> None:
    image_height, image_width = 200, 300
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.rectangle(mask, (30, 40), (90, 120), 255, -1)
    cv2.circle(mask, (220, 125), 28, 255, -1)
    geometry = GridGeometry(width_m=6.0, height_m=4.0, nx=60, ny=40)

    occupancy = mask_to_occupancy(mask, geometry)
    assert occupancy.dtype == np.bool_
    assert occupancy.shape == (40, 60)
    # Pixel-to-grid locations corresponding to object centers must be occupied.
    assert occupancy[16, 12]
    assert occupancy[25, 44]
    assert not occupancy[5, 30]


def test_metric_inflation_uses_conservative_cell_radii() -> None:
    geometry = GridGeometry(width_m=5.0, height_m=3.0, nx=51, ny=31)
    occupancy = np.zeros(geometry.shape_yx, dtype=bool)
    occupancy[15, 25] = True
    requested_radius = 0.21
    inflated, (radius_x, radius_y) = inflate_occupancy(occupancy, geometry, requested_radius)

    assert radius_x == 3  # ceil(0.21 / 0.10)
    assert radius_y == 3
    rows, cols = np.where(inflated)
    assert rows.min() == 15 - radius_y
    assert rows.max() == 15 + radius_y
    assert cols.min() == 25 - radius_x
    assert cols.max() == 25 + radius_x
    assert np.count_nonzero(inflated) > np.count_nonzero(occupancy)


def test_products_preserve_true_means_occupied() -> None:
    geometry = GridGeometry(width_m=3.0, height_m=2.0, nx=31, ny=21)
    mask = np.zeros((210, 310), dtype=np.uint8)
    cv2.rectangle(mask, (100, 70), (160, 140), 255, -1)
    products = compute_occupancy_products(
        mask,
        geometry,
        robot_radius_m=0.1,
        perception_margin_m=0.05,
    )
    assert products.uninflated.dtype == np.bool_
    assert products.inflated.dtype == np.bool_
    assert np.all(products.inflated[products.uninflated])
    assert products.diagnostics["occupied_cells_inflated"] >= products.diagnostics["occupied_cells_uninflated"]
