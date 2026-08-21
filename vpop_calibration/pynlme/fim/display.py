import pandas as pd
from IPython.display import display
from typing import Any

good_threshold = 30.0
acceptable_threshold = 50.0


def color_rse(value: Any) -> str:
    """Color code a relative standard error against the usual thresholds."""
    if pd.isna(value):
        return ""
    if value < good_threshold:
        return "color: green"
    if value < acceptable_threshold:
        return "color: orange"
    return "color: red"


def show_table(df: pd.DataFrame, title: str) -> pd.DataFrame:
    print(title)
    display(df)
    return df


def display_show_summary(
    df: pd.DataFrame, title: str = "Summary of Population Parameters"
):
    print(title)
    display(df.style.map(color_rse, subset=["RSE (%)"]))
    return df
