"""Guidance vector field generation via Laplace solves."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .normals import signed_distance_from_occupancy, normal_field_from_signed_distance
from ..solvers.sor import solve_poisson_sor
from ..solvers.sparse_direct import solve_poisson_sparse_direct
from ..solvers.conjugate_gradient import solve_poisson_cg


def build_boundary_flux_strength(
    shape: tuple[int, ...],
    base_flux_strength: float,
    nonuniform_axis: str | None = None,
    nonuniform_gain: float = 0.0,
) -> np.ndarray:
    """Build a scalar boundary flux strength field."""
    strength = np.full(shape, float(base_flux_strength), dtype=float)
    if nonuniform_axis is None or nonuniform_gain == 0.0:
        return strength
    axes = {"x": 0, "y": 1, "z": 2}
    axis = axes.get(nonuniform_axis.lower())
    if axis is None or axis >= len(shape):
        return strength
    coords = np.linspace(0.0, 1.0, shape[axis])
    view = [None] * len(shape)
    view[axis] = slice(None)
    factor = 1.0 + nonuniform_gain * coords[tuple(view)]
    return strength * factor


def solve_component(
    rhs: np.ndarray,
    solve_mask: np.ndarray,
    boundary_values: np.ndarray,
    grid_spacing: tuple[float, ...],
    solver: str,
    solver_options: Dict[str, Any],
) -> tuple[np.ndarray, dict]:
    """Solve one Laplace/Poisson component using the selected solver."""
    if solver == "sor":
        return solve_poisson_sor(rhs, solve_mask, boundary_values, grid_spacing, **solver_options)
    if solver == "sparse_direct":
        return solve_poisson_sparse_direct(rhs, solve_mask, boundary_values, grid_spacing)
    if solver == "conjugate_gradient":
        return solve_poisson_cg(rhs, solve_mask, boundary_values, grid_spacing, **solver_options)
    raise ValueError(f"Unsupported solver: {solver}")


def build_guidance_vector_field(
    occupancy_mask: np.ndarray,
    solve_mask: np.ndarray,
    boundary_mask: np.ndarray,
    grid_spacing: tuple[float, ...],
    solver: str,
    solver_options: Dict[str, Any],
    base_flux_strength: float = 0.5,
    nonuniform_axis: str | None = None,
    nonuniform_gain: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Construct a smooth guidance vector field.

    The boundary vector is set to b(y)n(y), where n is the approximate outward
    normal from occupied space into free space. Each component is extended into
    the free domain by solving Laplace's equation Δv_i=0.
    """
    dim = occupancy_mask.ndim
    signed = signed_distance_from_occupancy(occupancy_mask, grid_spacing)
    normals = normal_field_from_signed_distance(signed, grid_spacing)
    strength = build_boundary_flux_strength(occupancy_mask.shape, base_flux_strength, nonuniform_axis, nonuniform_gain)
    boundary_vec = normals * strength[..., None]

    zero_rhs = np.zeros_like(signed, dtype=float)
    components = []
    info = {"component_solver_info": []}
    for axis in range(dim):
        boundary_values = np.zeros_like(signed, dtype=float)
        boundary_values[boundary_mask] = boundary_vec[..., axis][boundary_mask]
        comp, comp_info = solve_component(zero_rhs, solve_mask, boundary_values, grid_spacing, solver, solver_options)
        components.append(comp)
        info["component_solver_info"].append(comp_info)
    vector_field = np.stack(components, axis=-1)
    info.update({
        "signed_distance": signed,
        "normals": normals,
        "boundary_vector": boundary_vec,
        "base_flux_strength": float(base_flux_strength),
    })
    return vector_field, info
