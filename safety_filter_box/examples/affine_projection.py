"""Project a nominal input onto two affine safe half-spaces."""
from __future__ import annotations

import numpy as np
from safety_box_core import AffineConstraint, ConstraintBundle, DecisionLayout
from safety_filter_box import MultiCertificateFilter, SafetyFilterConfig

layout = DecisionLayout.from_sizes(control=2)
bundle = ConstraintBundle(
    source_box="example",
    constraints=(
        AffineConstraint(np.array([[1.0, 0.0]]), np.array([0.0]), "x_nonnegative", "example", "x_ge_0"),
        AffineConstraint(np.array([[0.0, 1.0]]), np.array([0.0]), "y_nonnegative", "example", "y_ge_0"),
    ),
)
filter_box = MultiCertificateFilter(
    SafetyFilterConfig(solver="hildreth"),
    layout=layout,
    lower=np.array([-2.0, -2.0]),
    upper=np.array([2.0, 2.0]),
)
result = filter_box.solve(np.array([-1.0, 0.5]), bundles=(bundle,))
print("status:", result.status.value)
print("safe decision:", result.decision)
print("minimum residual:", float(np.min(result.residuals)))
