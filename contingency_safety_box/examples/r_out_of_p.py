"""Evaluate a 2-out-of-3 combinatorial certificate."""
from __future__ import annotations

import numpy as np
from safety_box_core import CertificateEvaluation, DecisionLayout
from contingency_safety_box import ContingencyBox, ContingencyBoxConfig

certificates = tuple(
    CertificateEvaluation(
        identifier=f"LZ{index}",
        value=value,
        drift=-0.1 * value,
        control_gradient=np.array([1.0, 0.0]),
        source="clf_roa",
    )
    for index, value in enumerate((0.4, 0.1, -0.2))
)
box = ContingencyBox(ContingencyBoxConfig(required_certified=2))
evaluation = box.evaluate(certificates)
layout = DecisionLayout.from_sizes(control=2, omega_contingency=1)
bundle = box.build_constraints(certificates, layout=layout, evaluation=evaluation)
print("pivot:", evaluation.pivot)
print("certified count:", evaluation.certified_count)
print("critical IDs:", evaluation.critical_ids)
print("constraint rows:", len(bundle.constraints))
