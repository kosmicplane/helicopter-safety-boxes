r"""Quadratic CLF synthesis from LQR and a continuous Lyapunov equation.

For the error dynamics

.. math::

    \dot e = A e + B u, \qquad u=-K e,

this module constructs

.. math::

    A_{\mathrm{cl}} = A-BK,
    \qquad
    A_{\mathrm{cl}}^\top P + P A_{\mathrm{cl}} = -Q_L,
    \qquad
    V(e)=e^\top P e.

The resulting artifact contains every matrix and numerical diagnostic required
for an auditable landing-zone certificate.
r"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

from safety_box_core import EquilibriumTarget


@dataclass(frozen=True, slots=True)
class QuadraticCLFArtifact:
    r"""Persistable quadratic CLF and certified sublevel-set metadata.r"""

    target: EquilibriumTarget
    A: np.ndarray
    B: np.ndarray
    K: np.ndarray
    P: np.ndarray
    Q_lyapunov: np.ndarray
    c: float
    certification_method: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("A", "B", "K", "P", "Q_lyapunov"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinity.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not np.isfinite(self.c) or self.c <= 0.0:
            raise ValueError("The ROA level c must be finite and positive.")
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    def save(self, directory: str | Path) -> None:
        r"""Save matrix data as NPZ and human-readable metadata as JSON.r"""

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        stem = output_directory / self.target.identifier

        np.savez_compressed(
            stem.with_suffix(".npz"),
            x_star=self.target.x_star,
            u_star=self.target.u_star,
            A=self.A,
            B=self.B,
            K=self.K,
            P=self.P,
            Q_lyapunov=self.Q_lyapunov,
            c=np.array(self.c),
        )
        payload = {
            "target_id": self.target.identifier,
            "x_star": self.target.x_star.tolist(),
            "u_star": self.target.u_star.tolist(),
            "c": float(self.c),
            "certification_method": self.certification_method,
            "diagnostics": _jsonable(self.diagnostics),
        }
        stem.with_suffix(".json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def analytic_input_feasible_c(
    P: np.ndarray,
    K: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    r"""Largest ellipsoid level compatible with componentwise input bounds.

    For row ``k_i`` of ``K`` and ellipsoid ``e.T P e <= c``, support-function
    maximization gives

    .. math::

        \max |k_i e| = \sqrt{c\,k_iP^{-1}k_i^\top}.

    The returned level is the minimum componentwise bound.  This calculation
    is exact for the linear feedback and symmetric zero-centered input limits.
    r"""

    matrix_P = np.asarray(P, dtype=float)
    matrix_K = np.asarray(K, dtype=float)
    if matrix_P.shape[0] != matrix_P.shape[1]:
        raise ValueError("P must be square.")
    if matrix_K.shape[1] != matrix_P.shape[0]:
        raise ValueError("K and P dimensions are inconsistent.")

    inverse_P = np.linalg.inv(matrix_P)
    lower_bound = np.asarray(lower, dtype=float).reshape(matrix_K.shape[0])
    upper_bound = np.asarray(upper, dtype=float).reshape(matrix_K.shape[0])
    levels: list[float] = []

    for row, lo, hi in zip(matrix_K, lower_bound, upper_bound, strict=True):
        support = float(row @ inverse_P @ row.T)
        available = min(abs(float(lo)), abs(float(hi)))
        if support > 0.0:
            levels.append(available**2 / support)

    return float(min(levels)) if levels else float("inf")


def construct_quadratic_clf(
    *,
    model: object,
    target: EquilibriumTarget,
    lqr_q: np.ndarray,
    lqr_r: np.ndarray,
    q_lyapunov: np.ndarray,
    control_lower: np.ndarray,
    control_upper: np.ndarray,
    roa_fraction: float = 0.9,
    manual_gain: np.ndarray | None = None,
    manual_c: float | None = None,
) -> QuadraticCLFArtifact:
    r"""Construct and verify one local quadratic CLF artifact.r"""

    if not 0.0 < roa_fraction <= 1.0:
        raise ValueError("roa_fraction must lie in (0, 1].")

    A, B = model.linearize(target.x_star, target.u_star)  # type: ignore[attr-defined]
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    q_lqr = np.asarray(lqr_q, dtype=float)
    r_lqr = np.asarray(lqr_r, dtype=float)
    q_lyap = np.asarray(q_lyapunov, dtype=float)

    if manual_gain is None:
        riccati = solve_continuous_are(A, B, q_lqr, r_lqr)
        K = np.linalg.solve(r_lqr, B.T @ riccati)
        constructor = "quadratic_lqr"
    else:
        K = np.asarray(manual_gain, dtype=float)
        constructor = "quadratic_user_gain"

    A_cl = A - B @ K
    closed_loop_eigenvalues = np.linalg.eigvals(A_cl)
    if np.max(np.real(closed_loop_eigenvalues)) >= -1.0e-10:
        raise ValueError(
            f"A-BK is not Hurwitz for {target.identifier}: "
            f"{closed_loop_eigenvalues}"
        )

    # SciPy solves A X + X A^T = Q. Passing A_cl.T and -Q_L yields the
    # continuous Lyapunov equation A_cl.T P + P A_cl = -Q_L.
    P = solve_continuous_lyapunov(A_cl.T, -q_lyap)
    P = 0.5 * (P + P.T)
    P_eigenvalues = np.linalg.eigvalsh(P)
    if np.min(P_eigenvalues) <= 0.0:
        raise ValueError("The Lyapunov solution P is not positive definite.")

    residual = A_cl.T @ P + P @ A_cl + q_lyap
    c_max = analytic_input_feasible_c(
        P,
        K,
        np.asarray(control_lower, dtype=float),
        np.asarray(control_upper, dtype=float),
    )
    c = float(manual_c) if manual_c is not None else float(roa_fraction) * c_max
    certification_method = (
        "manual_verified" if manual_c is not None else "analytic_linear_input_bound"
    )

    diagnostics = {
        "constructor": constructor,
        "closed_loop_eigenvalues_real": np.real(closed_loop_eigenvalues),
        "closed_loop_eigenvalues_imag": np.imag(closed_loop_eigenvalues),
        "P_eigenvalues": P_eigenvalues,
        "P_condition_number": float(np.linalg.cond(P)),
        "lyapunov_residual_fro": float(np.linalg.norm(residual, "fro")),
        "input_feasible_c_max": float(c_max),
        "roa_fraction": float(roa_fraction),
    }
    return QuadraticCLFArtifact(
        target=target,
        A=A,
        B=B,
        K=K,
        P=P,
        Q_lyapunov=q_lyap,
        c=c,
        certification_method=certification_method,
        diagnostics=diagnostics,
    )
