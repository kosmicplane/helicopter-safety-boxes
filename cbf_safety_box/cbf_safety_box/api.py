"""High-level public API for the CBF Safety Box.

The standard entry point, :meth:`CBFBox.filter_control`, constructs a CBF or
HOCBF row from one ``SafetySample``.  The contingency landing study additionally
uses :meth:`CBFBox.filter_affine_constraints`, which solves the same minimally
invasive QP after multiple already-built affine safety/stability rows are added.
This keeps all optimization inside the CBF box instead of hiding a second QP in
the runtime script.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import CBFBoxConfig
from .constraints.backstepping import compute_backstepping_value
from .constraints.builders import Constraint, build_constraints
from .constraints.control_limits import bounds_from_config
from .diagnostics.metrics import command_metrics, hocbf_metrics
from .optimization.closed_form import solve_halfspace_projection
from .optimization.cvxpy_qp import solve_qp_cvxpy
from .optimization.feasibility import check_feasibility
from .optimization.qp_problem import QPProblem
from .optimization.scipy_qp import solve_qp_scipy
from .result import CBFBoxResult
from .safety_data.sample import SafetySample
from .state import SystemState


class CBFBox:
    """High-level safety filter that maps a nominal decision to a safe one."""

    def __init__(self, config: CBFBoxConfig):
        """Store and validate an immutable-style user configuration."""
        self.config = config
        self.config.validate()

    @property
    def enabled(self) -> bool:
        """Whether the environmental certificate provider is active."""
        return bool(getattr(self.config, "enabled", True))

    def build_constraint(
        self,
        *,
        state: Any,
        sample: SafetySample,
        layout: Any,
        control_block: str = "control",
    ) -> Any:
        r"""Build one hard environmental row in a shared decision layout.

        This interoperability API complements :meth:`filter_control`; it does
        not replace the original full CBF Safety Box.  The original box may
        still solve standalone velocity, acceleration-HOCBF, backstepping, and
        multi-row projection problems.  Integrated applications can instead
        ask it to construct only its environmental constraint and pass that row
        to ``safety_filter_box`` together with CLF and contingency rows.
        """
        try:
            from safety_box_core import AffineConstraint, ConstraintHardness
        except ImportError as exc:  # pragma: no cover - only for standalone installs
            raise ImportError(
                "The shared-decision provider API requires safety_box_core. "
                "The standalone filter_control API remains available without it."
            ) from exc

        gradient_norm = float(np.linalg.norm(sample.grad_h))
        if gradient_norm <= float(self.config.minimum_gradient_norm):
            raise ValueError("Safety gradient is too small to define a reliable CBF half-space.")
        effective = SafetySample(
            h=float(sample.h) - float(self.config.h_margin),
            grad_h=np.asarray(sample.grad_h, dtype=float),
            hessian_h=None if sample.hessian_h is None else np.asarray(sample.hessian_h, dtype=float),
            laplacian_h=sample.laplacian_h,
            partial_h_t=sample.partial_h_t,
            metadata={**sample.metadata, "raw_h": float(sample.h), "h_margin": float(self.config.h_margin)},
        )
        if hasattr(state, "x"):
            state_array = np.asarray(state.x, dtype=float).reshape(-1)
            spatial_dimension = effective.dimension
            system_state = SystemState(
                position=state_array[:spatial_dimension],
                velocity=state_array[spatial_dimension:2 * spatial_dimension]
                if state_array.size >= 2 * spatial_dimension else None,
                time=float(getattr(state, "time_s", 0.0)),
            )
        elif isinstance(state, SystemState):
            system_state = state
        else:
            state_array = np.asarray(state, dtype=float).reshape(-1)
            spatial_dimension = effective.dimension
            system_state = SystemState(
                position=state_array[:spatial_dimension],
                velocity=state_array[spatial_dimension:2 * spatial_dimension]
                if state_array.size >= 2 * spatial_dimension else None,
            )
        local = build_constraints(self.config, system_state, effective)[0]
        row = np.zeros((local.A.shape[0], int(layout.dimension)), dtype=float)
        block = layout.block_slice(control_block)
        if block.stop - block.start != local.A.shape[1]:
            raise ValueError("CBF control dimension does not match the decision-layout control block.")
        row[:, block] = local.A
        metadata = {
            **local.metadata,
            "raw_h": float(sample.h),
            "h_effective": float(effective.h),
            "gradient_norm": gradient_norm,
            "partial_h_t": 0.0 if sample.partial_h_t is None else float(sample.partial_h_t),
        }
        # Static constraints ignore partial_h_t.  For a time-varying field,
        # move it to the right-hand side using h_dot = ... + partial_h_t.
        bound = np.asarray(local.b, dtype=float).copy()
        if sample.partial_h_t is not None:
            bound -= float(sample.partial_h_t)
        return AffineConstraint(
            A=row,
            b=bound,
            name=local.name,
            source_box="cbf_safety_box",
            equation_id="CBF-RD1" if self.config.mode == "velocity" else "HOCBF-RD2",
            hardness=ConstraintHardness.HARD,
            priority=0,
            metadata=metadata,
        )

    def evaluate(self, state: Any, context: Mapping[str, Any] | None = None) -> Any:
        """Constraint-provider protocol used by integrated experiments."""
        from safety_box_core import BoxStatus, ConstraintBundle

        if not self.enabled:
            return ConstraintBundle.disabled("cbf_safety_box")
        if context is None:
            return ConstraintBundle(
                status=BoxStatus.INVALID,
                source_box="cbf_safety_box",
                message="CBF context was not supplied.",
            )
        try:
            constraint = self.build_constraint(
                state=state,
                sample=context["sample"],
                layout=context["layout"],
                control_block=context.get("control_block", "control"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            return ConstraintBundle(
                constraints=(),
                status=BoxStatus.INVALID,
                source_box="cbf_safety_box",
                message=str(exc),
            )
        return ConstraintBundle(
            constraints=(constraint,),
            status=BoxStatus.READY,
            source_box="cbf_safety_box",
        )

    def filter_control(self, state: SystemState, safety: SafetySample, u_nom: np.ndarray) -> CBFBoxResult:
        """Filter a nominal physical control through the configured CBF/HOCBF.

        This method is the normal single-certificate API.  It deliberately calls
        ``filter_affine_constraints`` so every path uses the same QP backend and
        feasibility diagnostics.
        """
        nominal = np.asarray(u_nom, dtype=float).reshape(-1)
        if nominal.size != safety.dimension:
            raise ValueError("u_nom dimension must match safety.grad_h dimension.")

        constraints = build_constraints(self.config, state, safety)
        lower, upper = bounds_from_config(
            self.config.control_lower_bound,
            self.config.control_upper_bound,
            nominal.size,
        )
        result = self.filter_affine_constraints(
            decision_nominal=nominal,
            constraints=constraints,
            lower_bounds=lower,
            upper_bounds=upper,
        )

        # Add diagnostics that depend on the semantic CBF mode rather than only
        # on the generic affine QP representation.
        if self.config.mode == "acceleration" and state.velocity is not None and safety.hessian_h is not None:
            result.diagnostics["hocbf_metrics"] = hocbf_metrics(
                safety.h,
                safety.grad_h,
                state.velocity,
                safety.hessian_h,
                self.config.alpha1,
            )
            result.hocbf_residual = result.cbf_residual
        if self.config.mode == "backstepping" and state.velocity is not None:
            result.diagnostics["backstepping"] = compute_backstepping_value(
                h=safety.h,
                velocity=state.velocity,
                grad_h=safety.grad_h,
                mu=self.config.backstepping.mu,
                k1_type=self.config.backstepping.k1_type,
                k1_gain=self.config.backstepping.k1_gain,
                nominal_velocity=nominal,
            )
        return result

    def filter_affine_constraints(
        self,
        *,
        decision_nominal: np.ndarray,
        constraints: Sequence[Constraint],
        lower_bounds: np.ndarray | Sequence[float] | None = None,
        upper_bounds: np.ndarray | Sequence[float] | None = None,
        quadratic_weights: np.ndarray | Sequence[float] | None = None,
        norm_bound_indices: list[list[int]] | None = None,
        norm_bound_values: list[float] | None = None,
        use_slack: bool | None = None,
    ) -> CBFBoxResult:
        """Solve a generic multi-row CBF/CLF projection QP inside the box.

        Parameters are intentionally explicit.  The runtime may build mission
        certificates, but it cannot solve or bypass the safety QP itself.
        """
        nominal = np.asarray(decision_nominal, dtype=float).reshape(-1)
        if not constraints:
            raise ValueError("At least one affine constraint is required.")

        # Validate that every constraint acts on the same augmented decision.
        matrices = []
        vectors = []
        for constraint in constraints:
            matrix = np.asarray(constraint.A, dtype=float)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.shape[1] != nominal.size:
                raise ValueError(
                    f"Constraint {constraint.name!r} has {matrix.shape[1]} columns; "
                    f"the decision has dimension {nominal.size}."
                )
            matrices.append(matrix)
            vectors.append(np.asarray(constraint.b, dtype=float).reshape(-1))

        A = np.vstack(matrices)
        b = np.concatenate(vectors)
        lower = None if lower_bounds is None else np.asarray(lower_bounds, dtype=float).reshape(-1)
        upper = None if upper_bounds is None else np.asarray(upper_bounds, dtype=float).reshape(-1)
        weights = None if quadratic_weights is None else np.asarray(quadratic_weights, dtype=float).reshape(-1)
        slack_enabled = self.config.use_slack if use_slack is None else bool(use_slack)
        norm_groups = norm_bound_indices or []
        norm_values = norm_bound_values or []

        # The closed-form backend is exact only for one unweighted affine row and
        # no norm constraint.  All contingency QPs use the SciPy/CVXPY backend.
        can_use_closed_form = (
            self.config.solver == "closed_form"
            and A.shape[0] == 1
            and not slack_enabled
            and not norm_groups
            and (weights is None or np.allclose(weights, 1.0))
        )
        if can_use_closed_form:
            raw = solve_halfspace_projection(
                nominal,
                A[0],
                b[0],
                lower_bounds=lower,
                upper_bounds=upper,
                tolerance=self.config.diagnostics.numerical_tolerance,
            )
            raw.setdefault("residuals", np.array([raw.get("residual", np.nan)], dtype=float))
            raw.setdefault("norm_residuals", np.array([], dtype=float))
            raw.setdefault("slack", 0.0)
            raw.setdefault("iterations", 0)
        else:
            problem = QPProblem(
                u_nom=nominal,
                A_ineq=A,
                b_ineq=b,
                lower_bounds=lower,
                upper_bounds=upper,
                quadratic_weights=weights,
                norm_bound_indices=norm_groups,
                norm_bound_values=norm_values,
                use_slack=slack_enabled,
                slack_weight=self.config.slack_weight,
                metadata={"constraint_names": [constraint.name for constraint in constraints]},
            )
            if self.config.solver == "cvxpy":
                raw = solve_qp_cvxpy(problem)
            else:
                raw = solve_qp_scipy(problem, tolerance=self.config.diagnostics.numerical_tolerance)

        safe_decision = np.asarray(raw["u_safe"], dtype=float).reshape(-1)
        residuals = A @ safe_decision - b
        minimum_residual = float(np.min(residuals)) if residuals.size else None
        feasibility = check_feasibility(
            safe_decision,
            A,
            b,
            lower,
            upper,
            self.config.diagnostics.numerical_tolerance,
            norm_groups,
            norm_values,
        )

        diagnostics = {
            "constraints": [constraint.name for constraint in constraints],
            "constraint_metadata": [constraint.metadata for constraint in constraints],
            "feasibility": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in feasibility.items()
            },
            "command_metrics": command_metrics(nominal, safe_decision),
            "quadratic_weights": np.ones(nominal.size).tolist() if weights is None else weights.tolist(),
            "norm_bound_indices": norm_groups,
            "norm_bound_values": norm_values,
            "emergency_slack": float(raw.get("slack", 0.0)),
            "iterations": int(raw.get("iterations", 0)),
            "raw_norm_residuals": np.asarray(raw.get("norm_residuals", []), dtype=float).tolist(),
        }

        return CBFBoxResult(
            u_safe=safe_decision,
            u_nom=nominal,
            was_filtered=bool(raw.get("was_filtered", np.linalg.norm(safe_decision - nominal) > 1.0e-8)),
            cbf_residual=minimum_residual,
            hocbf_residual=None,
            constraint_matrix=A,
            constraint_vector=b,
            solver_status=str(raw.get("status", "unknown")),
            solve_time=float(raw.get("solve_time", 0.0)),
            active_constraints=list(raw.get("active_constraints", [])),
            diagnostics=diagnostics,
        )
