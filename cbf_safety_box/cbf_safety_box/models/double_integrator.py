"""Double-integrator model helper: p_dot = v, v_dot = a."""

def expected_control_dimension(position_dimension: int) -> int:
    """For v_dot=a, the acceleration command dimension equals position dimension."""
    return int(position_dimension)
