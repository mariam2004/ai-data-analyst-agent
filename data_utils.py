"""
File loading, validation, and dataset profiling utilities.

Only compact schema/profile information should ever be sent to the LLM —
never the full DataFrame.
"""

from __future__ import annotations

import io
import warnings
from typing import Any

import pandas as pd


class DataLoadError(Exception):
    """Raised when a user-uploaded file cannot be loaded or is invalid."""


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV file from raw bytes into a DataFrame."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The CSV file is malformed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface as a friendly error
        raise DataLoadError(f"Could not read CSV file: {exc}") from exc
    return df


def load_excel(file_bytes: bytes) -> pd.DataFrame:
    """Load an Excel (.xlsx/.xls) file from raw bytes into a DataFrame."""
    try:
        df = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Could not read Excel file: {exc}") from exc
    return df


def load_file(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """Dispatch to the correct loader based on file extension."""
    name = filename.lower()
    if name.endswith(".csv"):
        df = load_csv(file_bytes)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = load_excel(file_bytes)
    else:
        raise DataLoadError(
            "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."
        )
    validate_dataframe(df)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise DataLoadError if the DataFrame is unusable."""
    if df is None or df.empty:
        raise DataLoadError("The uploaded file contains no data.")
    if len(df.columns) == 0:
        raise DataLoadError("The uploaded file has no columns.")


def profile_data(df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact, LLM-friendly profile of the dataset.

    This is intentionally lightweight: only schema-level and summary
    statistics are included, never raw row data.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # Try to detect string columns that are actually dates but stored as text.
    for col in list(categorical_cols):
        if _looks_like_date(df[col]):
            datetime_cols.append(col)
            categorical_cols.remove(col)

    missing = df.isna().sum()
    missing_dict = {col: int(missing[col]) for col in df.columns if missing[col] > 0}

    basic_stats: dict[str, dict[str, float]] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        for col, stats in desc.items():
            basic_stats[col] = {k: round(float(v), 4) for k, v in stats.items()}

    profile = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": missing_dict,
        "duplicate_count": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "basic_statistics": basic_stats,
    }
    return profile


def _looks_like_date(series: pd.Series, sample_size: int = 20) -> bool:
    """Heuristically check whether a text column holds parseable dates."""
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
    except Exception:  # noqa: BLE001
        return False
    return parsed.notna().mean() > 0.8


def profile_to_prompt_text(profile: dict[str, Any]) -> str:
    """Render the profile dict as compact text for LLM prompts."""
    lines = [
        f"rows={profile['row_count']}, columns={profile['column_count']}",
        f"columns: {profile['columns']}",
        f"dtypes: {profile['dtypes']}",
        f"missing_values: {profile['missing_values'] or 'none'}",
        f"duplicate_rows: {profile['duplicate_count']}",
        f"numeric_columns: {profile['numeric_columns']}",
        f"categorical_columns: {profile['categorical_columns']}",
        f"datetime_columns: {profile['datetime_columns']}",
    ]
    return "\n".join(lines)
