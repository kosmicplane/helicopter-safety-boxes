"""Solver backends for the unified multi-certificate optimization.

The portable default is a warm-started SciPy SLSQP backend.  Solver-specific
logic is isolated behind a small protocol so an OSQP or Clarabel adapter can be
added without modifying any safety box or experiment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


@dataclass(frozen=True, slots=True)
class QPData:
    """Dense representation of a strictly convex affine-constrained QP."""

    nominal: np.ndarray
    weights: np.ndarray
    A: np.ndarray
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True)
class RawSolverResult:
    """Backend-independent numerical result before explicit verification."""

    x: np.ndarray
    status: str
    success: bool
    solve_time_s: float
    iterations: int
    message: str


class QPSolver(Protocol):
    """Minimal interface implemented by optimization backends."""

    name: str

    def solve(self, data: QPData) -> RawSolverResult: ...


class ScipySLSQPSolver:
    """Warm-started portable backend for small dense QPs."""

    name = "scipy_slsqp"

    def __init__(self, tolerance: float = 1.0e-9, max_iterations: int = 200) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self._warm_start: np.ndarray | None = None

    def solve(self, data: QPData) -> RawSolverResult:
        nominal = np.asarray(data.nominal, dtype=float)
        weights = np.asarray(data.weights, dtype=float)
        if np.any(weights <= 0.0):
            raise ValueError("All quadratic weights must be strictly positive.")

        if self._warm_start is None or self._warm_start.shape != nominal.shape:
            initial = nominal.copy()
        else:
            initial = self._warm_start.copy()
        initial = np.clip(initial, data.lower, data.upper)

        def objective(decision: np.ndarray) -> float:
            error = decision - nominal
            return 0.5 * float(np.dot(weights * error, error))

        def gradient(decision: np.ndarray) -> np.ndarray:
            return weights * (decision - nominal)

        constraints: list[LinearConstraint] = []
        if data.A.size:
            constraints.append(
                LinearConstraint(
                    data.A,
                    data.b,
                    np.full(data.b.shape, np.inf, dtype=float),
                )
            )

        started = perf_counter()
        result = minimize(
            objective,
            initial,
            jac=gradient,
            bounds=Bounds(data.lower, data.upper),
            constraints=constraints,
            method="SLSQP",
            options={
                "ftol": self.tolerance,
                "maxiter": self.max_iterations,
                "disp": False,
            },
        )
        elapsed = perf_counter() - started
        decision = np.asarray(result.x, dtype=float)
        if np.all(np.isfinite(decision)):
            self._warm_start = decision.copy()

        return RawSolverResult(
            x=decision,
            status="optimal" if result.success else "solver_warning",
            success=bool(result.success),
            solve_time_s=elapsed,
            iterations=int(getattr(result, "nit", 0)),
            message=str(result.message),
        )


_SOLVER_REGISTRY: dict[str, type] = {
    "scipy": ScipySLSQPSolver,
    "slsqp": ScipySLSQPSolver,
    "scipy_slsqp": ScipySLSQPSolver,
}


def make_solver(name: str, **kwargs: object) -> QPSolver:
    """Construct a registered solver backend by name."""

    normalized = str(name).lower()
    try:
        constructor = _SOLVER_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_SOLVER_REGISTRY))
        if normalized in {"osqp", "clarabel"}:
            raise RuntimeError(
                f"The {normalized} adapter is not installed in this portable bundle. "
                f"Use scipy_slsqp or add an adapter implementing QPSolver."
            ) from exc
        raise ValueError(f"Unsupported solver {name!r}; choose {supported}.") from exc
    return constructor(**kwargs)  # type: ignore[arg-type]

class HildrethQPSolver:
    r"""Warm-started dual coordinate solver for diagonal projection QPs.

    It solves

    .. math::
        \min_z \tfrac12(z-z_0)^T W(z-z_0)\quad\text{s.t.}\quad Az\ge b,

    after representing finite component bounds as additional affine rows.
    The backend is dependency-light, deterministic, and particularly effective
    for the small dense certificate filters used in this repository.
    """

    name = "hildreth"

    def __init__(
        self,
        tolerance: float = 1.0e-8,
        max_iterations: int = 4000,
        relaxation: float = 1.15,
    ) -> None:
        if tolerance <= 0.0 or max_iterations < 1:
            raise ValueError("tolerance and max_iterations must be positive.")
        if not 0.0 < relaxation <= 2.0:
            raise ValueError("relaxation must lie in (0, 2].")
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.relaxation = float(relaxation)
        self._multipliers: np.ndarray | None = None
        self._row_count = -1

    @staticmethod
    def _append_bounds(data: QPData) -> tuple[np.ndarray, np.ndarray]:
        rows = [np.asarray(data.A, dtype=float)] if data.A.size else []
        rhs = [np.asarray(data.b, dtype=float)] if data.b.size else []
        dimension = data.nominal.size
        for index, value in enumerate(np.asarray(data.lower, dtype=float)):
            if np.isfinite(value):
                row = np.zeros(dimension); row[index] = 1.0
                rows.append(row.reshape(1, -1)); rhs.append(np.array([value]))
        for index, value in enumerate(np.asarray(data.upper, dtype=float)):
            if np.isfinite(value):
                row = np.zeros(dimension); row[index] = -1.0
                rows.append(row.reshape(1, -1)); rhs.append(np.array([-value]))
        if not rows:
            return np.empty((0, dimension)), np.empty(0)
        return np.vstack(rows), np.concatenate(rhs)

    def solve(self, data: QPData) -> RawSolverResult:
        started = perf_counter()
        nominal = np.asarray(data.nominal, dtype=float).reshape(-1)
        weights = np.asarray(data.weights, dtype=float).reshape(-1)
        if np.any(weights <= 0.0):
            raise ValueError("All quadratic weights must be strictly positive.")
        A, b = self._append_bounds(data)
        if A.shape[0] == 0:
            return RawSolverResult(nominal.copy(), "optimal", True, perf_counter()-started, 0, "no constraints")
        nominal_residual = A @ nominal - b
        if float(np.min(nominal_residual)) >= -self.tolerance:
            self._multipliers = np.zeros(A.shape[0])
            self._row_count = A.shape[0]
            return RawSolverResult(nominal.copy(), "optimal", True, perf_counter()-started, 0, "nominal feasible")
        inverse_weights = 1.0 / weights
        M = (A * inverse_weights[None, :]) @ A.T
        q = b - A @ nominal
        diagonal = np.diag(M)
        valid_rows = diagonal > 1.0e-16
        if not np.all(valid_rows):
            # Zero rows can only be feasible when their right-hand side is nonpositive.
            if np.any(b[~valid_rows] > self.tolerance):
                return RawSolverResult(nominal.copy(), "invalid", False, perf_counter()-started, 0, "infeasible zero-norm row")
            A = A[valid_rows]; b = b[valid_rows]
            M = M[np.ix_(valid_rows, valid_rows)]
            q = q[valid_rows]; diagonal = diagonal[valid_rows]
        if self._multipliers is not None and self._row_count == A.shape[0]:
            multipliers = self._multipliers.copy()
        else:
            multipliers = np.zeros(A.shape[0], dtype=float)
        converged = False
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            maximum_change = 0.0
            for row in range(A.shape[0]):
                gradient = float(M[row] @ multipliers - q[row])
                candidate = max(
                    0.0,
                    multipliers[row]
                    - self.relaxation * gradient / diagonal[row],
                )
                maximum_change = max(maximum_change, abs(candidate - multipliers[row]))
                multipliers[row] = candidate
            if maximum_change <= self.tolerance:
                decision = nominal + inverse_weights * (A.T @ multipliers)
                if float(np.min(A @ decision - b)) >= -10.0 * self.tolerance:
                    converged = True
                    break
        decision = nominal + inverse_weights * (A.T @ multipliers)
        residual = A @ decision - b
        feasible = bool(float(np.min(residual)) >= -10.0 * self.tolerance)
        self._multipliers = multipliers.copy()
        self._row_count = A.shape[0]
        return RawSolverResult(
            x=decision,
            status="optimal" if feasible else "max_iterations",
            success=feasible,
            solve_time_s=perf_counter() - started,
            iterations=iterations,
            message=(
                "dual coordinate solver converged"
                if converged
                else ("feasible iterate at iteration limit" if feasible else f"minimum residual {float(np.min(residual)):.3e}")
            ),
        )


_SOLVER_REGISTRY.update(
    {
        "hildreth": HildrethQPSolver,
        "dual_coordinate": HildrethQPSolver,
    }
)
