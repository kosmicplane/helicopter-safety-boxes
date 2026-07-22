"""Slack-variable helpers for soft CBF constraints.

Slack support is disabled by default.  It is useful for debugging infeasible
problems, but strict safety-critical controllers should avoid slack unless they
also define how violations are handled.
"""

def slack_penalty(slack: float, weight: float) -> float:
    """Quadratic slack penalty."""
    return float(weight) * float(slack) ** 2
