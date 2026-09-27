
from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)


# ============================================================
# COLUMN INTELLIGENCE
# ============================================================

METADATA_KEYWORDS = {
    "_input",
    "_num",
    "_widget",
    "_source",
    "_result",
    "_pageurl",
    "_page_url",
    "_sourceurl",
    "_source_url",
}

URL_KEYWORDS = {
    "url",
    "uri",
    "link",
    "website",
    "web_url",
    "pageurl",
    "page_url",
}

IDENTIFIER_EXACT = {
    "id",
    "rank",
    "index",
    "row_id",
    "row_number",
    "row_num",
    "record_id",
    "result_number",
    "resultnumber",
}

IDENTIFIER_SUFFIXES = (
    "_id",
    "_key",
)

IDENTIFIER_PREFIXES = (
    "id_",
    "key_",
)

IDENTIFIER_KEYWORDS = {
    "identifier",
    "rowid",
    "recordid",
}


def _normalize_column_name(column: Any) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _is_metadata_column(column: Any) -> bool:
    name = _normalize_column_name(column)

    if name.startswith("_"):
        return True

    if name in METADATA_KEYWORDS:
        return True

    for keyword in METADATA_KEYWORDS:
        if keyword in name:
            return True

    if any(keyword in name for keyword in URL_KEYWORDS):
        return True

    return False


def _is_identifier_column(
    df: pd.DataFrame,
    column: Any,
) -> bool:

    name = _normalize_column_name(column)

    # Strong name-based identifier rules.
    if name in IDENTIFIER_EXACT:
        return True

    if name in IDENTIFIER_KEYWORDS:
        return True

    if name.endswith(IDENTIFIER_SUFFIXES):
        return True

    if name.startswith(IDENTIFIER_PREFIXES):
        return True

    # Numeric columns should NOT be classified as identifiers
    # merely because most values are unique. Metrics such as
    # revenue, growth, salary, age, etc. can naturally be unique.
    #
    # Use uniqueness only as a weak signal for non-numeric text
    # columns.
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            unique_ratio = (
                df[column].nunique(dropna=True)
                / max(len(df), 1)
            )

            if unique_ratio >= 0.98 and len(df) >= 20:
                return True

    except Exception:
        pass

    return False


def _detect_datetime_columns(
    df: pd.DataFrame,
) -> list[str]:

    datetime_cols = []

    date_name_keywords = (
        "date",
        "time",
        "timestamp",
        "datetime",
        "year",
        "month",
    )

    for col in df.columns:

        # Never treat metadata or identifiers as dates.
        if _is_metadata_column(col):
            continue

        if _is_identifier_column(df, col):
            continue

        series = df[col]

        # Already a real datetime dtype.
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # Only attempt automatic parsing on text columns.
        if series.dtype != "object":
            continue

        non_null = series.dropna()

        if non_null.empty:
            continue

        name = _normalize_column_name(col)

        # Strong hint from the column name.
        name_looks_like_date = any(
            keyword in name
            for keyword in date_name_keywords
        )

        try:
            # "mixed" avoids the Pandas format-inference warning
            # and handles multiple date formats safely.
            converted = pd.to_datetime(
                non_null,
                errors="coerce",
                format="mixed",
            )

            valid_ratio = converted.notna().mean()

            # Use a stricter threshold for generic detection.
            if valid_ratio >= 0.85:
                if name_looks_like_date or valid_ratio >= 0.95:
                    datetime_cols.append(col)

        except (TypeError, ValueError):
            continue

    return datetime_cols


def _classify_columns(
    df: pd.DataFrame,
) -> dict[str, list[str]]:

    datetime_cols = _detect_datetime_columns(df)

    metadata_cols = []
    identifier_cols = []
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:

        if col in datetime_cols:
            continue

        if _is_metadata_column(col):
            metadata_cols.append(col)
            continue

        if _is_identifier_column(df, col):
            identifier_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
        "metadata": metadata_cols,
        "identifier": identifier_cols,
    }


# ============================================================
# GENERAL HELPERS
# ============================================================

def _safe_text(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _number(value: float) -> str:
    return f"{value:,.2f}"


def _table_style() -> TableStyle:
    return TableStyle(
        [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1f2937"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold",
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey,
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#f9fafb"),
                ],
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP",
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ]
    )


def _make_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> BytesIO:

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.bar(
        df[x].astype(str),
        df[y],
    )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer


def _make_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> BytesIO:

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.plot(
        df[x],
        df[y],
        marker="o",
    )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer


def _make_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
) -> BytesIO:

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.scatter(
        df[x],
        df[y],
        alpha=0.7,
    )

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    plt.tight_layout()

    buffer = BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    buffer.seek(0)

    return buffer


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf_report(
    df: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
) -> bytes:

    if df is None or df.empty:
        raise ValueError(
            "Cannot generate a report from an empty dataset."
        )

    working_df = df.copy()

    groups = _classify_columns(working_df)

    numeric_cols = groups["numeric"]
    categorical_cols = groups["categorical"]
    datetime_cols = groups["datetime"]
    metadata_cols = groups["metadata"]
    identifier_cols = groups["identifier"]

    rows = len(working_df)
    columns = len(working_df.columns)

    missing_values = int(
        working_df.isna().sum().sum()
    )

    duplicate_rows = int(
        working_df.duplicated().sum()
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="AI Data Analyst Report",
        author="AI Data Analyst Agent",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=26,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=14,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=15,
        leading=18,
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=13,
        spaceAfter=6,
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AI Data Analyst Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            _safe_text(dataset_name),
            subtitle_style,
        )
    )

    # ========================================================
    # 1. EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            section_style,
        )
    )

    summary = (
        f"The dataset contains <b>{rows:,}</b> rows and "
        f"<b>{columns:,}</b> columns. "
        f"It includes <b>{len(numeric_cols)}</b> analytical numeric "
        f"columns, <b>{len(categorical_cols)}</b> categorical dimensions, "
        f"and <b>{len(datetime_cols)}</b> date/time columns."
    )

    if metadata_cols:
        summary += (
            f" {len(metadata_cols)} metadata columns were detected "
            "and excluded from analytical calculations."
        )

    if identifier_cols:
        summary += (
            f" {len(identifier_cols)} identifier columns were detected "
            "and excluded from aggregation statistics."
        )

    if missing_values:
        summary += (
            f" The dataset contains {missing_values:,} missing values."
        )
    else:
        summary += " No missing values were detected."

    story.append(
        Paragraph(
            summary,
            body_style,
        )
    )

    # ========================================================
    # 2. DATASET OVERVIEW
    # ========================================================

    story.append(
        Paragraph(
            "2. Dataset Overview",
            section_style,
        )
    )

    overview_rows = [
        ["Metric", "Value"],
        ["Rows", f"{rows:,}"],
        ["Columns", f"{columns:,}"],
        ["Analytical Numeric Columns", f"{len(numeric_cols):,}"],
        ["Categorical Dimensions", f"{len(categorical_cols):,}"],
        ["Date/Time Columns", f"{len(datetime_cols):,}"],
        ["Metadata Columns", f"{len(metadata_cols):,}"],
        ["Identifier Columns", f"{len(identifier_cols):,}"],
        ["Missing Values", f"{missing_values:,}"],
        ["Duplicate Rows", f"{duplicate_rows:,}"],
    ]

    overview_table = Table(
        overview_rows,
        colWidths=[
            3.5 * inch,
            1.7 * inch,
        ],
    )

    overview_table.setStyle(_table_style())

    story.append(overview_table)

    # ========================================================
    # 3. KEY NUMERIC METRICS
    # ========================================================

    if numeric_cols:

        story.append(
            Paragraph(
                "3. Key Numeric Metrics",
                section_style,
            )
        )

        metric_rows = [
            [
                "Metric",
                "Mean",
                "Median",
                "Min",
                "Max",
            ]
        ]

        for col in numeric_cols[:10]:

            series = pd.to_numeric(
                working_df[col],
                errors="coerce",
            ).dropna()

            if series.empty:
                continue

            metric_rows.append(
                [
                    _safe_text(col),
                    _number(float(series.mean())),
                    _number(float(series.median())),
                    _number(float(series.min())),
                    _number(float(series.max())),
                ]
            )

        if len(metric_rows) > 1:

            metric_table = Table(
                metric_rows,
                colWidths=[
                    1.7 * inch,
                    1.05 * inch,
                    1.05 * inch,
                    1.05 * inch,
                    1.05 * inch,
                ],
            )

            metric_table.setStyle(_table_style())

            story.append(metric_table)

    # ========================================================
    # 4. TOP CATEGORICAL VALUES
    # ========================================================

    useful_categorical = [
        col
        for col in categorical_cols
        if 2 <= working_df[col].nunique(dropna=True) <= 30
    ]

    if useful_categorical:

        story.append(
            Paragraph(
                "4. Categorical Analysis",
                section_style,
            )
        )

        for col in useful_categorical[:4]:

            counts = (
                working_df[col]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(10)
                .reset_index()
            )

            counts.columns = [
                col,
                "Count",
            ]

            story.append(
                Paragraph(
                    f"<b>{_safe_text(col)}</b>",
                    body_style,
                )
            )

            rows_table = [
                [
                    "Value",
                    "Count",
                ]
            ]

            for _, row in counts.iterrows():

                rows_table.append(
                    [
                        Paragraph(
                            _safe_text(row[col]),
                            body_style,
                        ),
                        f"{int(row['Count']):,}",
                    ]
                )

            table = Table(
                rows_table,
                colWidths=[
                    4.3 * inch,
                    1.2 * inch,
                ],
            )

            table.setStyle(_table_style())

            story.append(table)
            story.append(Spacer(1, 7))

    # ========================================================
    # 5. TIME ANALYSIS
    # ========================================================

    if datetime_cols and numeric_cols:

        date_col = datetime_cols[0]
        value_col = numeric_cols[0]

        dates = pd.to_datetime(
            working_df[date_col],
            errors="coerce",
        )

        values = pd.to_numeric(
            working_df[value_col],
            errors="coerce",
        )

        time_df = pd.DataFrame(
            {
                "_date": dates,
                "_value": values,
            }
        ).dropna()

        if not time_df.empty:

            monthly = (
                time_df
                .assign(
                    _period=time_df["_date"].dt.to_period("M")
                )
                .groupby("_period")["_value"]
                .sum()
                .reset_index()
            )

            if len(monthly) >= 2:

                monthly["_date"] = (
                    monthly["_period"]
                    .dt.to_timestamp()
                )

                story.append(
                    Paragraph(
                        "5. Time-Based Analysis",
                        section_style,
                    )
                )

                story.append(
                    Paragraph(
                        f"Monthly trend of "
                        f"<b>{_safe_text(value_col)}</b> "
                        f"using <b>{_safe_text(date_col)}</b>.",
                        body_style,
                    )
                )

                chart = _make_line_chart(
                    monthly,
                    "_date",
                    "_value",
                    f"{value_col} Over Time",
                )

                story.append(
                    Image(
                        chart,
                        width=6.6 * inch,
                        height=3.7 * inch,
                    )
                )

    # ========================================================
    # 6. CATEGORY VS METRIC
    # ========================================================

    if useful_categorical and numeric_cols:

        category_col = useful_categorical[0]
        metric_col = numeric_cols[0]

        grouped = (
            working_df
            .groupby(
                category_col,
                dropna=False,
            )[metric_col]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
            .reset_index()
        )

        if len(grouped) >= 2:

            story.append(
                Paragraph(
                    "6. Category vs Numeric Analysis",
                    section_style,
                )
            )

            story.append(
                Paragraph(
                    f"Aggregated "
                    f"<b>{_safe_text(metric_col)}</b> "
                    f"by "
                    f"<b>{_safe_text(category_col)}</b>.",
                    body_style,
                )
            )

            chart = _make_bar_chart(
                grouped,
                category_col,
                metric_col,
                f"{metric_col} by {category_col}",
            )

            story.append(
                Image(
                    chart,
                    width=6.6 * inch,
                    height=3.7 * inch,
                )
            )

    # ========================================================
    # 7. NUMERIC RELATIONSHIP
    # ========================================================

    if len(numeric_cols) >= 2:

        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        scatter_df = working_df[
            [x_col, y_col]
        ].copy()

        scatter_df[x_col] = pd.to_numeric(
            scatter_df[x_col],
            errors="coerce",
        )

        scatter_df[y_col] = pd.to_numeric(
            scatter_df[y_col],
            errors="coerce",
        )

        scatter_df = scatter_df.dropna()

        if len(scatter_df) >= 3:

            story.append(
                Paragraph(
                    "7. Numeric Relationship",
                    section_style,
                )
            )

            story.append(
                Paragraph(
                    f"Relationship between "
                    f"<b>{_safe_text(x_col)}</b> and "
                    f"<b>{_safe_text(y_col)}</b>.",
                    body_style,
                )
            )

            chart = _make_scatter_chart(
                scatter_df.head(2000),
                x_col,
                y_col,
                f"{x_col} vs {y_col}",
            )

            story.append(
                Image(
                    chart,
                    width=6.6 * inch,
                    height=3.7 * inch,
                )
            )

    # ========================================================
    # 8. DATA QUALITY
    # ========================================================

    story.append(
        Paragraph(
            "8. Data Quality",
            section_style,
        )
    )

    missing_by_column = (
        working_df
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_columns = (
        missing_by_column[
            missing_by_column > 0
        ]
    )

    quality_rows = [
        ["Check", "Result"],
        [
            "Missing values",
            f"{missing_values:,}",
        ],
        [
            "Columns with missing data",
            f"{len(missing_columns):,}",
        ],
        [
            "Duplicate rows",
            f"{duplicate_rows:,}",
        ],
        [
            "Metadata columns excluded",
            f"{len(metadata_cols):,}",
        ],
        [
            "Identifier columns excluded",
            f"{len(identifier_cols):,}",
        ],
    ]

    quality_table = Table(
        quality_rows,
        colWidths=[
            3.3 * inch,
            1.7 * inch,
        ],
    )

    quality_table.setStyle(_table_style())

    story.append(quality_table)

    # --------------------------------------------------------
    # Missing values by column
    # --------------------------------------------------------

    if not missing_columns.empty:

        story.append(
            Spacer(1, 10)
        )

        story.append(
            Paragraph(
                "Missing Values by Column",
                ParagraphStyle(
                    "MissingTitle",
                    parent=body_style,
                    fontSize=11,
                    leading=14,
                    spaceBefore=6,
                    spaceAfter=6,
                ),
            )
        )

        missing_rows = [
            [
                "Column",
                "Missing",
                "Missing %",
            ]
        ]

        # Show all missing columns when there are only a few.
        # Otherwise show the top 15 columns by missing count.
        missing_display = (
            missing_columns
            .sort_values(ascending=False)
        )

        if len(missing_display) > 15:
            missing_display = missing_display.head(15)

        for col, count in missing_display.items():

            percentage = (
                float(count) / max(rows, 1)
            ) * 100

            missing_rows.append(
                [
                    Paragraph(
                        _safe_text(col),
                        body_style,
                    ),
                    f"{int(count):,}",
                    f"{percentage:.2f}%",
                ]
            )

        missing_table = Table(
            missing_rows,
            colWidths=[
                3.5 * inch,
                1.0 * inch,
                1.0 * inch,
            ],
            repeatRows=1,
        )

        missing_table.setStyle(
            _table_style()
        )

        story.append(
            missing_table
        )

        if len(missing_columns) > 15:
            story.append(
                Paragraph(
                    f"Showing the 15 columns with the highest "
                    f"number of missing values out of "
                    f"{len(missing_columns)} affected columns.",
                    ParagraphStyle(
                        "MissingNote",
                        parent=body_style,
                        fontSize=8,
                        textColor=colors.grey,
                    ),
                )
            )

    else:

        story.append(
            Paragraph(
                "No missing values were detected in any column.",
                body_style,
            )
        )

    # ========================================================
    # 9. COLUMN CLASSIFICATION
    # ========================================================

    story.append(
        Paragraph(
            "9. Detected Column Types",
            section_style,
        )
    )

    classification_rows = [
        [
            "Column",
            "Detected Type",
        ]
    ]

    for col in working_df.columns:

        if col in numeric_cols:
            detected = "Analytical Numeric"

        elif col in categorical_cols:
            detected = "Categorical / Dimension"

        elif col in datetime_cols:
            detected = "Date / Time"

        elif col in metadata_cols:
            detected = "Metadata"

        elif col in identifier_cols:
            detected = "Identifier"

        else:
            detected = "Other"

        classification_rows.append(
            [
                Paragraph(
                    _safe_text(col),
                    body_style,
                ),
                detected,
            ]
        )

    classification_table = Table(
        classification_rows,
        colWidths=[
            4.1 * inch,
            1.7 * inch,
        ],
        repeatRows=1,
    )

    classification_table.setStyle(
        _table_style()
    )

    story.append(
        classification_table
    )

    # ========================================================
    # FOOTER
    # ========================================================

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Generated by AI Data Analyst Agent",
            ParagraphStyle(
                "Footer",
                parent=body_style,
                alignment=TA_CENTER,
                fontSize=8,
                textColor=colors.grey,
            ),
        )
    )

    document.build(story)

    buffer.seek(0)

    return buffer.getvalue()
