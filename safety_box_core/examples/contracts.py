"""Demonstrate canonical shared contracts."""
from __future__ import annotations

import numpy as np
from safety_box_core import AffineConstraint, DecisionLayout

layout = DecisionLayout.from_sizes(control=2, omega_contingency=1)
row = layout.lift_row(np.array([1.0, -0.5]), "control")
row[layout.scalar_index("omega_contingency")] = 0.2
constraint = AffineConstraint(
    A=row.reshape(1, -1),
    b=np.array([-0.1]),
    name="example_row",
    source_box="example",
    equation_id="A_z_ge_b",
)
print("layout:", layout.blocks)
print("residual:", constraint.residual(np.array([0.3, 0.1, 0.0])))
