"""Configuration dataclasses for the CBF Safety Box.

The configuration is intentionally independent of ROS, PX4, Gazebo, or any
specific simulator.  It only describes how the CBF constraints and QP solver
should behave.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BacksteppingConfig:
    """Options for the experimental relative-degree-2 backstepping helper."""

    mu: float = 1.0
    k1_type: str = "gradient_ascent"  # gradient_ascent, nominal_tracking, zero
    k1_gain: float = 1.0


@dataclass
class DiagnosticsConfig:
    """Numerical diagnostics options."""

    check_feasibility: bool = True
    numerical_tolerance: float = 1.0e-8


@dataclass
class PlottingConfig:
    """Plotting options for optional visualization helpers."""

    enabled: bool = True
    save: bool = True
    show: bool = False
    dpi: int = 180


@dataclass
class CBFBoxConfig:
    """Configuration for the high-level CBFBox API.

    Parameters
    ----------
    mode:
        Which safety-filter mode to use: ``velocity``, ``acceleration``, or
        ``backstepping``.
    solver:
        Which optimizer to use: ``closed_form``, ``scipy``, or ``cvxpy``.
    alpha, alpha1, alpha2:
        CBF/HOCBF gains.  ``alpha`` is used for relative-degree-1 velocity CBFs;
        ``alpha1`` and ``alpha2`` are used for relative-degree-2 HOCBFs.
    control_lower_bound, control_upper_bound:
        Optional control limits.  They are applied component-wise.
    use_slack:
        Enables a nonnegative slack variable for scipy/cvxpy QPs.  Disabled by
        default because strict safety filtering is usually preferred.
    """

    enabled: bool = True
    mode: str = "velocity"
    solver: str = "closed_form"

    alpha: float = 3.0
    alpha1: float = 2.0
    alpha2: float = 2.0
    gamma_type: str = "linear"
    gamma_gain: float = 1.0
    h_margin: float = 0.0
    minimum_gradient_norm: float = 1.0e-9

    control_lower_bound: list[float] | None = None
    control_upper_bound: list[float] | None = None

    use_slack: bool = False
    slack_weight: float = 1.0e4

    backstepping: BacksteppingConfig = field(default_factory=BacksteppingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)


    @property
    def gamma1(self) -> float:
        """Alias for ``alpha1`` used by integrated HOCBF experiments."""
        return float(self.alpha1)

    @property
    def gamma2(self) -> float:
        """Alias for ``alpha2`` used by integrated HOCBF experiments."""
        return float(self.alpha2)

    @property
    def velocity_gamma(self) -> float:
        """Alias for the relative-degree-one CBF gain ``alpha``."""
        return float(self.alpha)

    def validate(self) -> None:
        """Validate user-facing configuration values."""
        if self.mode not in {"velocity", "acceleration", "backstepping"}:
            raise ValueError(f"Unsupported CBF mode: {self.mode!r}")
        if self.solver not in {"closed_form", "scipy", "cvxpy"}:
            raise ValueError(f"Unsupported QP solver: {self.solver!r}")
        if self.gamma_type != "linear":
            raise ValueError("Only linear gamma functions are implemented in this box.")
        if self.alpha <= 0 or self.alpha1 <= 0 or self.alpha2 <= 0:
            raise ValueError("CBF gains alpha, alpha1, and alpha2 must be positive.")
        if self.slack_weight <= 0:
            raise ValueError("slack_weight must be positive.")
        if self.h_margin < 0.0:
            raise ValueError("h_margin must be nonnegative.")
        if self.minimum_gradient_norm <= 0.0:
            raise ValueError("minimum_gradient_norm must be positive.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CBFBoxConfig":
        """Load a configuration from a YAML file."""
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CBFBoxConfig":
        """Create a configuration from a dictionary.

        Nested dictionaries named ``backstepping``, ``diagnostics``, and
        ``plotting`` are converted into their corresponding dataclasses.
        """
        data = dict(data)
        # Integrated experiments use explicit CBF/HOCBF names and gamma notation.
        # Map those aliases to the original public API without breaking existing
        # users of ``mode=velocity|acceleration`` and ``alpha*``.
        mode_aliases = {"velocity_cbf": "velocity", "acceleration_hocbf": "acceleration"}
        if "mode" in data:
            data["mode"] = mode_aliases.get(str(data["mode"]), data["mode"])
        if "gamma1" in data:
            data["alpha1"] = data.pop("gamma1")
        if "gamma2" in data:
            data["alpha2"] = data.pop("gamma2")
        if "velocity_gamma" in data:
            data["alpha"] = data.pop("velocity_gamma")
        back = data.pop("backstepping", {}) or {}
        diag = data.pop("diagnostics", {}) or {}
        plot = data.pop("plotting", {}) or {}
        cfg = cls(
            **data,
            backstepping=BacksteppingConfig(**back),
            diagnostics=DiagnosticsConfig(**diag),
            plotting=PlottingConfig(**plot),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly dictionary representation."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "solver": self.solver,
            "alpha": self.alpha,
            "alpha1": self.alpha1,
            "alpha2": self.alpha2,
            "gamma_type": self.gamma_type,
            "gamma_gain": self.gamma_gain,
            "h_margin": self.h_margin,
            "minimum_gradient_norm": self.minimum_gradient_norm,
            "control_lower_bound": self.control_lower_bound,
            "control_upper_bound": self.control_upper_bound,
            "use_slack": self.use_slack,
            "slack_weight": self.slack_weight,
            "backstepping": self.backstepping.__dict__,
            "diagnostics": self.diagnostics.__dict__,
            "plotting": self.plotting.__dict__,
        }
