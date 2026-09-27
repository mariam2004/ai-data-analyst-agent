"""
Analytical tools available to the agent.

The LLM never computes numbers itself — it only chooses which of these
tools to call. All tools here perform the actual work with DuckDB, Pandas,
or Plotly and return structured, factual results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_utils import profile_data, profile_to_prompt_text

# ---------------------------------------------------------------------------
# SQL safety
# ---------------------------------------------------------------------------

_BLOCKED_KEYWORDS = [
    "drop", "delete", "update", "insert", "alter", "create", "attach",
    "detach", "copy", "pragma", "export", "import", "install", "load",
    "call", "vacuum", "replace", "grant", "revoke",
]


class UnsafeSQLError(Exception):
    """Raised when a generated SQL query is not a safe, read-only SELECT."""


def validate_sql(sql: str) -> str:
    """Ensure the SQL is a single, read-only SELECT statement."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Empty SQL query.")

    lowered = cleaned.lower()
    if ";" in cleaned:
        raise UnsafeSQLError("Multiple SQL statements are not allowed.")
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise UnsafeSQLError("Only read-only SELECT queries are allowed.")

    for keyword in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", lowered):
            raise UnsafeSQLError(f"Query contains a disallowed keyword: {keyword}")

    return cleaned


# ---------------------------------------------------------------------------
# A. Data profile tool
# ---------------------------------------------------------------------------

@dataclass
class ProfileResult:
    profile: dict[str, Any]
    text: str


def run_profile_tool(df: pd.DataFrame) -> ProfileResult:
    """Return a compact structured description of the current dataset."""
    profile = profile_data(df)
    return ProfileResult(profile=profile, text=profile_to_prompt_text(profile))


# ---------------------------------------------------------------------------
# B. DuckDB SQL tool
# ---------------------------------------------------------------------------

@dataclass
class SQLResult:
    sql: str
    result_df: pd.DataFrame
    row_count: int
    error: Optional[str] = None


def run_sql_tool(df: pd.DataFrame, sql: str) -> SQLResult:
    """Run a validated, read-only SQL query against `df` (exposed as `data`)."""
    try:
        clean_sql = validate_sql(sql)
    except UnsafeSQLError as exc:
        return SQLResult(sql=sql, result_df=pd.DataFrame(), row_count=0, error=str(exc))

    try:
        con = duckdb.connect(database=":memory:")
        con.register("data", df)
        result_df = con.execute(clean_sql).fetchdf()
        con.close()
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent for retry
        return SQLResult(sql=clean_sql, result_df=pd.DataFrame(), row_count=0, error=str(exc))

    return SQLResult(sql=clean_sql, result_df=result_df, row_count=len(result_df))


# ---------------------------------------------------------------------------
# C. Pandas analysis tool
# ---------------------------------------------------------------------------

@dataclass
class PandasResult:
    operation: str
    result: Any
    error: Optional[str] = None


def pandas_describe(df: pd.DataFrame, columns: Optional[list[str]] = None) -> PandasResult:
    """Descriptive statistics for numeric columns."""
    try:
        subset = df[columns] if columns else df.select_dtypes(include="number")
        return PandasResult(operation="describe", result=subset.describe().to_dict())
    except Exception as exc:  # noqa: BLE001
        return PandasResult(operation="describe", result=None, error=str(exc))


def pandas_correlation(df: pd.DataFrame, columns: Optional[list[str]] = None) -> PandasResult:
    """Correlation matrix between numeric columns."""
    try:
        subset = df[columns] if columns else df.select_dtypes(include="number")
        return PandasResult(operation="correlation", result=subset.corr().round(4).to_dict())
    except Exception as exc:  # noqa: BLE001
        return PandasResult(operation="correlation", result=None, error=str(exc))


def pandas_missing_analysis(df: pd.DataFrame) -> PandasResult:
    """Missing-value counts and percentages per column."""
    try:
        missing = df.isna().sum()
        pct = (missing / len(df) * 100).round(2)
        result = {
            col: {"missing_count": int(missing[col]), "missing_pct": float(pct[col])}
            for col in df.columns
            if missing[col] > 0
        }
        return PandasResult(operation="missing_analysis", result=result)
    except Exception as exc:  # noqa: BLE001
        return PandasResult(operation="missing_analysis", result=None, error=str(exc))


def pandas_period_comparison(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period_a: tuple[str, str],
    period_b: tuple[str, str],
    breakdown_col: Optional[str] = None,
) -> PandasResult:
    """Compare a metric between two date periods, optionally broken down by
    a dimension. Used for diagnostic "why did X change" questions.

    period_a is treated as the "current"/observed period, period_b as the
    comparison/baseline period. Each period is a (start, end) ISO date pair.
    """
    try:
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")

        a_start, a_end = pd.to_datetime(period_a[0]), pd.to_datetime(period_a[1])
        b_start, b_end = pd.to_datetime(period_b[0]), pd.to_datetime(period_b[1])

        mask_a = (work[date_col] >= a_start) & (work[date_col] <= a_end)
        mask_b = (work[date_col] >= b_start) & (work[date_col] <= b_end)

        total_a = float(work.loc[mask_a, value_col].sum())
        total_b = float(work.loc[mask_b, value_col].sum())
        abs_change = total_a - total_b
        pct_change = (abs_change / total_b * 100) if total_b else None

        breakdown = None
        if breakdown_col:
            grp_a = work.loc[mask_a].groupby(breakdown_col)[value_col].sum()
            grp_b = work.loc[mask_b].groupby(breakdown_col)[value_col].sum()
            combined = pd.DataFrame({"period_a": grp_a, "period_b": grp_b}).fillna(0)
            combined["abs_change"] = combined["period_a"] - combined["period_b"]
            combined["pct_change"] = combined.apply(
                lambda r: (r["abs_change"] / r["period_b"] * 100) if r["period_b"] else None,
                axis=1,
            )
            combined = combined.sort_values("abs_change")
            breakdown = combined.round(2).to_dict(orient="index")

        result = {
            "period_a_total": round(total_a, 2),
            "period_b_total": round(total_b, 2),
            "absolute_change": round(abs_change, 2),
            "percent_change": round(pct_change, 2) if pct_change is not None else None,
            "breakdown_by": breakdown_col,
            "breakdown": breakdown,
        }
        return PandasResult(operation="period_comparison", result=result)
    except Exception as exc:  # noqa: BLE001
        return PandasResult(operation="period_comparison", result=None, error=str(exc))


# ---------------------------------------------------------------------------
# D. Plotly visualization tool
# ---------------------------------------------------------------------------

@dataclass
class ChartResult:
    figure: Optional[go.Figure]
    chart_type: str
    error: Optional[str] = None


def build_chart(
    data: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
) -> ChartResult:
    """Build a Plotly figure from already-validated result data.

    chart_type is one of: "line", "bar", "histogram", "scatter", "stacked_bar".
    """
    if data is None or data.empty:
        return ChartResult(figure=None, chart_type=chart_type, error="No data to plot.")

    try:
        if chart_type == "line":
            fig = px.line(data, x=x, y=y, color=color, title=title, markers=True)
        elif chart_type == "bar":
            fig = px.bar(data, x=x, y=y, color=color, title=title)
        elif chart_type == "stacked_bar":
            fig = px.bar(data, x=x, y=y, color=color, title=title, barmode="stack")
        elif chart_type == "histogram":
            fig = px.histogram(data, x=x, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(data, x=x, y=y, color=color, title=title)
        else:
            return ChartResult(
                figure=None, chart_type=chart_type, error=f"Unknown chart type: {chart_type}"
            )
    except Exception as exc:  # noqa: BLE001
        return ChartResult(figure=None, chart_type=chart_type, error=str(exc))

    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return ChartResult(figure=fig, chart_type=chart_type)


def suggest_chart_type(
    data: pd.DataFrame, date_col: Optional[str], category_col: Optional[str], numeric_cols: list[str]
) -> str:
    """Pick a sensible default chart type given the shape of the result."""
    if date_col:
        return "line"
    if category_col and len(numeric_cols) == 1:
        return "bar"
    if len(numeric_cols) >= 2:
        return "scatter"
    return "bar"
