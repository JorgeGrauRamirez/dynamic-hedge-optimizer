"""
Data loading and preprocessing.

The raw data comes pre-merged in `Merged_Shipping_Data.csv` (weekly frequency).
All prices are already scaled to thousands (USD '000 / day).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import pandas as pd


# These are the Baltic Index (paper / FFA) columns. Everything else in the
# dataset that is not `Date` is a physical route assessment.
FFA_COLUMNS_ALL: List[str] = [
    "Cape Avg 5TC",
    "Pmx Avg 5TC",
    "Pmx Avg 4TC",
    "Smx Avg 11TC",
    "Smx Avg 10TC",
    "Handysize Avg 7TC",
]


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the merged shipping dataset and return it sorted by date.

    The CSV is expected to contain a `Date` column plus physical route columns
    and FFA/Baltic index columns, all already scaled to thousands.
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").reset_index(drop=True)
    return df


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (physical_columns, ffa_columns) present in the dataframe.

    Only FFA columns from `FFA_COLUMNS_ALL` that are actually in `df` are
    returned, and everything else (except `Date`) is treated as physical.
    """
    ffa = [c for c in FFA_COLUMNS_ALL if c in df.columns]
    physical = [c for c in df.columns if c not in ffa and c != "Date"]
    return physical, ffa


def get_recommended_ffa_universe(df: pd.DataFrame) -> List[str]:
    """FFA columns with enough history to be useful for calibration.

    `Smx Avg 11TC` starts in 2023-05 so it has much less history than the
    other indices. We drop it from the default universe so calibration
    windows are comparable across dates.
    """
    _, ffa_present = split_columns(df)
    return [c for c in ffa_present if c != "Smx Avg 11TC"]


def build_working_dataset(
    df: pd.DataFrame,
    target_physical_route: str,
    ffa_columns: List[str],
) -> pd.DataFrame:
    """Subset the dataframe to the columns we need and trim to valid dates.

    Applies forward-fill on the selected columns, drops remaining NaNs, and
    truncates at the last date where the physical route is actually
    reported (so we don't extrapolate beyond the available physical data).
    """
    working_cols = [target_physical_route] + list(ffa_columns)
    df_working = df[["Date"] + working_cols].ffill().dropna().reset_index(drop=True)

    phys_mask = df[target_physical_route].notna()
    if phys_mask.any():
        last_valid = df.loc[phys_mask, "Date"].iloc[-1]
        df_working = df_working[df_working["Date"] <= last_valid].reset_index(drop=True)

    return df_working
