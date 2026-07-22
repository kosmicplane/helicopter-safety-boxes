"""Configuration dataclasses for the Poisson safety box."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class ConstantConfig:
    """Configuration for constant negative forcing f=-c."""

    c: float = 1.0


@dataclass
class DistanceConfig:
    """Configuration for distance-based forcing."""

    alpha: float = 0.5


@dataclass
class AverageFluxConfig:
    """Configuration for average-flux forcing."""

    b_bar: float = -1.0


@dataclass
class GuidanceConfig:
    """Configuration for guidance-vector forcing."""

    beta: float = 10.0
    base_flux_strength: float = 0.5
    target_mean_abs_scale: float = 0.35
    nonuniform_axis: Optional[str] = None
    nonuniform_gain: float = 0.0


@dataclass
class SORConfig:
    """Configuration for SOR iterations."""

    omega: float = 1.75
    max_iter: int = 4000
    tolerance: float = 1e-4
    residual_check_interval: int = 25
    warm_start: bool = False


@dataclass
class CGConfig:
    """Configuration for conjugate-gradient solves."""

    tolerance: float = 1e-6
    max_iter: int = 2000


@dataclass
class PoissonBoxConfig:
    """Main configuration for :class:`PoissonSafetyBox`.

    The configuration is deliberately independent of robot controllers. The
    only goal is to control how h is built from an occupancy matrix.
    """

    grid_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
    boundary_value: float = 0.0
    outer_boundary_as_dirichlet: bool = True

    forcing_method: str = "constant"  # constant, distance, average_flux, guidance
    solver: str = "sor"  # sor, sparse_direct, conjugate_gradient

    compute_gradient: bool = True
    compute_hessian: bool = True
    compute_laplacian_check: bool = True

    plot: bool = False
    save_outputs: bool = True

    constant: ConstantConfig = field(default_factory=ConstantConfig)
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    average_flux: AverageFluxConfig = field(default_factory=AverageFluxConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    sor: SORConfig = field(default_factory=SORConfig)
    conjugate_gradient: CGConfig = field(default_factory=CGConfig)

    def validate(self) -> None:
        """Validate common configuration mistakes."""
        if self.forcing_method not in {"constant", "distance", "average_flux", "guidance"}:
            raise ValueError(f"Unsupported forcing_method: {self.forcing_method}")
        if self.solver not in {"sor", "sparse_direct", "conjugate_gradient"}:
            raise ValueError(f"Unsupported solver: {self.solver}")
        if len(self.grid_spacing) not in {2, 3}:
            raise ValueError("grid_spacing must have length 2 or 3")
        if any(s <= 0 for s in self.grid_spacing):
            raise ValueError("grid_spacing values must be positive")
        if self.constant.c <= 0:
            raise ValueError("constant.c must be positive because forcing is f=-c")
        if self.average_flux.b_bar >= 0:
            raise ValueError("average_flux.b_bar must be negative")
        if not (0.0 < self.sor.omega < 2.0):
            raise ValueError("SOR omega must be in (0, 2)")

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable dictionary."""
        return asdict(self)

    @staticmethod
    def _merge_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
        """Recursively merge a dictionary into a dataclass instance."""
        for key, value in values.items():
            if not hasattr(obj, key):
                continue
            current = getattr(obj, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                PoissonBoxConfig._merge_dataclass(current, value)
            else:
                setattr(obj, key, value)
        return obj

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PoissonBoxConfig":
        """Load a configuration from a YAML file."""
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        cfg = cls()
        cls._merge_dataclass(cfg, data)
        cfg.validate()
        return cfg

    def save_yaml(self, path: str | Path) -> None:
        """Save this configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
