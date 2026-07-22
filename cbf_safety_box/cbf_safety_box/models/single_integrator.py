"""Single-integrator model helper: p_dot = v."""

def expected_control_dimension(position_dimension: int) -> int:
    """For p_dot=v, the velocity command dimension equals position dimension."""
    return int(position_dimension)
