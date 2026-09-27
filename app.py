"""
Streamlit UI for the AI Data Analyst Agent.

Run with: streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent import DataAnalystAgent, GeminiClient
from data_utils import DataLoadError, load_file, profile_data, profile_to_prompt_text
from memory import AgentMemory

load_dotenv()

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")

st.title("AI Data Analyst Agent")
st.caption("Upload your data and ask questions in natural language.")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "memory" not in st.session_state:
    st.session_state.memory = AgentMemory()
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {"role", "content", "df", "figure", "details"}
if "df" not in st.session_state:
    st.session_state.df = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "agent" not in st.session_state:
    st.session_state.agent = None


# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error(
        "GEMINI_API_KEY is not set. Add it to a `.env` file in the project "
        "root before using the chat (e.g. GEMINI_API_KEY=your_api_key_here)."
    )


# ---------------------------------------------------------------------------
# File upload + dataset overview
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()
        df = load_file(uploaded_file.name, file_bytes)
        profile = profile_data(df)

        # Only rebuild the agent when a new file is uploaded.
        if st.session_state.df is None or not st.session_state.df.equals(df):
            st.session_state.df = df
            st.session_state.profile = profile
            st.session_state.chat = []
            st.session_state.memory = AgentMemory()
            if api_key:
                gemini = GeminiClient(api_key=api_key)
                st.session_state.agent = DataAnalystAgent(
                    df=df,
                    profile_text=profile_to_prompt_text(profile),
                    gemini=gemini,
                    memory=st.session_state.memory,
                )
    except DataLoadError as exc:
        st.error(str(exc))
        st.session_state.df = None

if st.session_state.df is not None and st.session_state.profile is not None:
    profile = st.session_state.profile
    with st.expander("Dataset overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", profile["row_count"])
        col2.metric("Columns", profile["column_count"])
        col3.metric("Duplicate rows", profile["duplicate_count"])

        st.write("**Columns and types:**")
        st.json(profile["dtypes"])

        if profile["missing_values"]:
            st.write("**Missing values:**")
            st.json(profile["missing_values"])
        else:
            st.write("No missing values detected.")


# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Ask a question about your data")

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("df") is not None and not turn["df"].empty:
            st.dataframe(turn["df"], use_container_width=True)
        if turn.get("figure") is not None:
            st.plotly_chart(turn["figure"], use_container_width=True)
        if turn.get("details"):
            with st.expander("Analysis details"):
                st.write(f"**Tools used:** {', '.join(turn['details']['tools']) or 'none'}")
                if turn["details"].get("sql"):
                    st.code(turn["details"]["sql"], language="sql")
                if turn["details"].get("notes"):
                    st.write("**Notes:**", "; ".join(turn["details"]["notes"]))

question = st.chat_input("e.g. What are the top 5 products by revenue?")

if question:
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    elif not api_key:
        st.warning("Please set GEMINI_API_KEY in your .env file first.")
    elif st.session_state.agent is None:
        st.warning("The agent is not ready yet. Try re-uploading the file.")
    else:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result = st.session_state.agent.ask(question)
                except Exception as exc:  # noqa: BLE001
                    result = None
                    st.error(f"The agent could not complete this request: {exc}")

            if result is not None:
                st.markdown(result.answer)
                if result.result_df is not None and not result.result_df.empty:
                    st.dataframe(result.result_df, use_container_width=True)
                if result.chart_figure is not None:
                    st.plotly_chart(result.chart_figure, use_container_width=True)
                with st.expander("Analysis details"):
                    st.write(f"**Tools used:** {', '.join(result.tool_used) or 'none'}")
                    if result.sql:
                        st.code(result.sql, language="sql")
                    if result.notes:
                        st.write("**Notes:**", "; ".join(result.notes))

                st.session_state.chat.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "df": result.result_df,
                        "figure": result.chart_figure,
                        "details": {"tools": result.tool_used, "sql": result.sql, "notes": result.notes},
                    }
                )


# ============================================================
# PDF REPORT
# ============================================================

from report import generate_pdf_report

st.divider()

st.subheader("📄 Data Analyst Report")

st.caption(
    "Generate a generic PDF report based on the currently uploaded dataset."
)

current_df = globals().get("df")
current_uploaded_file = globals().get("uploaded_file")

if current_df is not None:

    if st.button(
        "Generate PDF Report",
        type="primary",
        use_container_width=True,
    ):

        try:
            dataset_name = (
                current_uploaded_file.name
                if current_uploaded_file is not None
                else "Uploaded Dataset"
            )

            with st.spinner("Generating PDF report..."):

                pdf_bytes = generate_pdf_report(
                    current_df,
                    dataset_name=dataset_name,
                )

            st.session_state["pdf_report"] = pdf_bytes
            st.session_state["pdf_report_name"] = (
                "AI_Data_Analyst_Report.pdf"
            )

            st.success(
                "✅ PDF report generated successfully."
            )

        except Exception as exc:
            st.error(
                f"Could not generate the PDF report: {exc}"
            )

    if "pdf_report" in st.session_state:

        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state["pdf_report"],
            file_name=st.session_state.get(
                "pdf_report_name",
                "AI_Data_Analyst_Report.pdf",
            ),
            mime="application/pdf",
            use_container_width=True,
        )

else:

    st.info(
        "Upload a CSV or Excel file first to generate a report."
    )
