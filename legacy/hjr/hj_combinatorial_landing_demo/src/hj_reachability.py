"""Hamilton-Jacobi reachability for the reduced single-integrator model.

Model
-----
    p_dot = u,       ||u||_2 <= v_max

For this isotropic model in a static obstacle map, the backward reach-avoid value
function can be represented exactly through the obstacle-aware geodesic distance
D_j(p) to landing target j:

    V_j(p, tau) = v_max * (-tau) - D_j(p),     tau <= 0.

The target is reachable iff V_j >= 0.  The geodesic distance is the viscosity
solution of the corresponding static Eikonal/Hamilton-Jacobi equation in free
space.  This is a reduced-order HJR demonstration, not a 6-DOF rotorcraft HJR
solver.  Its output is therefore interpreted as a velocity-level certificate that
can later be tracked by PX4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from skimage.graph import MCP_Geometric

from .scenario import LandingZone, World


@dataclass
class ReachabilityField:
    zone: LandingZone
    distance: np.ndarray
    grad_distance: np.ndarray
    finite_mask: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    max_speed: float
    rejected: bool = False

    def __post_init__(self) -> None:
        options = dict(bounds_error=False, fill_value=np.nan)
        finite_distance = self.distance.copy()
        max_finite = float(np.max(finite_distance[self.finite_mask])) if np.any(self.finite_mask) else 1e6
        finite_distance[~self.finite_mask] = max_finite + 4.0 * self.max_speed
        self._distance_interp = RegularGridInterpolator(self.axes, finite_distance, **options)
        self._grad_interp = [
            RegularGridInterpolator(self.axes, self.grad_distance[..., d], bounds_error=False, fill_value=0.0)
            for d in range(3)
        ]

    def sample(self, position: np.ndarray, tau: float) -> dict:
        """Return V, grad(V), and partial V / partial tau at one position."""
        point = np.asarray(position, dtype=float).reshape(1, 3)
        distance = float(self._distance_interp(point)[0])
        grad_d = np.array([float(interp(point)[0]) for interp in self._grad_interp], dtype=float)
        if self.rejected or not np.isfinite(distance):
            return {"V": -1e3, "grad_V": np.zeros(3), "partial_tau": -self.max_speed, "distance": np.inf}
        return {
            "V": float(self.max_speed * (-float(tau)) - distance),
            "grad_V": -grad_d,
            "partial_tau": -self.max_speed,
            "distance": distance,
        }

    def value_grid(self, tau: float) -> np.ndarray:
        out = self.max_speed * (-float(tau)) - self.distance
        out = np.where(self.finite_mask, out, -1e3)
        if self.rejected:
            out[:] = -1e3
        return out


def _target_seed_indices(world: World, zone: LandingZone, occupancy: np.ndarray) -> list[tuple[int, int, int]]:
    X, Y, Z = world.mesh
    radius = max(zone.radius, max(world.spacing) * 1.5)
    seed_mask = (
        (X - zone.center[0]) ** 2
        + (Y - zone.center[1]) ** 2
        + ((Z - zone.center[2]) / 0.75) ** 2
        <= radius**2
    ) & (~occupancy)
    seeds = [tuple(int(v) for v in row) for row in np.argwhere(seed_mask)]
    if not seeds:
        nearest = world.world_to_index(zone.center)
        if not occupancy[nearest]:
            seeds = [nearest]
    return seeds


def compute_reachability_fields(
    world: World,
    occupancy: np.ndarray,
    max_speed: float,
    rejected_indices: set[int] | None = None,
) -> list[ReachabilityField]:
    """Compute one obstacle-aware HJ/geodesic field for every landing zone."""
    rejected_indices = rejected_indices or set()
    # Unit traversal cost in free cells and infinite cost in obstacles.
    cost = np.ones(world.shape, dtype=float)
    cost[occupancy] = np.inf
    fields: list[ReachabilityField] = []
    for index, zone in enumerate(world.landing_zones):
        rejected = index in rejected_indices
        seeds = [] if rejected else _target_seed_indices(world, zone, occupancy)
        if not seeds:
            distance = np.full(world.shape, np.inf, dtype=float)
            finite = np.zeros(world.shape, dtype=bool)
            grad = np.zeros(world.shape + (3,), dtype=float)
        else:
            mcp = MCP_Geometric(cost, sampling=world.spacing, fully_connected=True)
            distance, _ = mcp.find_costs(starts=seeds)
            finite = np.isfinite(distance) & (~occupancy)
            # A finite extension avoids invalid arithmetic in np.gradient.  The
            # gradient is zeroed again outside the connected free component.
            max_finite = float(np.max(distance[finite])) if np.any(finite) else 0.0
            extended = distance.copy()
            extended[~finite] = max_finite + 4.0 * max(world.spacing)
            grads = np.gradient(extended, *world.spacing, edge_order=1)
            grad = np.stack(grads, axis=-1)
            grad[~finite] = 0.0
            # Normalize small numerical anisotropies of the Eikonal gradient.
            norm = np.linalg.norm(grad, axis=-1, keepdims=True)
            valid = norm[..., 0] > 1e-8
            grad[valid] /= norm[valid]
        fields.append(
            ReachabilityField(
                zone=zone,
                distance=distance,
                grad_distance=grad,
                finite_mask=finite,
                axes=world.axes,
                max_speed=float(max_speed),
                rejected=rejected,
            )
        )
    return fields


def rth_largest(values: np.ndarray, r: int) -> float:
    """Return the r-th largest scalar, matching the paper's pivot function."""
    values = np.asarray(values, dtype=float)
    if r < 1 or r > values.size:
        raise ValueError(f"r must be in [1, {values.size}], got {r}")
    return float(np.sort(values)[::-1][r - 1])
