"""Small logging helpers used by examples and integration scripts."""


def banner(text: str) -> None:
    """Print a readable section banner with a blank line before it."""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)
