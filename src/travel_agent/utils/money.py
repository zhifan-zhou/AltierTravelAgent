"""Money utility functions."""


def usd(value: float) -> float:
    """Format as USD rounded to 2 decimal places."""
    return round(value, 2)


def format_usd(value: float) -> str:
    """Format float as USD string."""
    return f"${value:,.2f}"
