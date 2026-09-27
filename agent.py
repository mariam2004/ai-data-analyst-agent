"""
LangGraph agent orchestrating the AI Data Analyst workflow.

Graph flow:

    START -> retrieve_memory -> understand_question -> decide_tool
          -> execute_tool -> validate_result -> generate_answer
          -> save_memory -> END

The LLM (Gemini) only understands, plans, chooses tools, and explains
results. All numbers come from tools.py (DuckDB / Pandas / Plotly).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, TypedDict

import pandas as pd
from google import genai
from langgraph.graph import END, StateGraph

import prompts
import tools
from data_utils import profile_to_prompt_text
from memory import AgentMemory

# Change the model here to switch Gemini versions across the whole app.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

MAX_SQL_RETRIES = 2


# ---------------------------------------------------------------------------
# Gemini wrapper
# ---------------------------------------------------------------------------

class GeminiClient:
    """Thin wrapper around the google-genai SDK used for all LLM calls."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, system_instruction: str, user_prompt: str) -> str:
        import random
        import time
        from google.genai import errors

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config={
                        "system_instruction": system_instruction,
                        "temperature": 0.2,
                    },
                )

                return (response.text or "").strip()

            except errors.APIError as exc:
                status_code = getattr(exc, "code", None)

                # Retry only transient backend/rate-limit errors.
                if status_code in (408, 429, 500, 502, 503, 504):
                    if attempt < max_retries - 1:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        print(
                            f"Gemini temporary error {status_code}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue

                raise




# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    question: str
    resolved_question: str
    memory_context: str
    tools_to_use: list[str]
    sql: str
    sql_error: Optional[str]
    sql_result: Optional[pd.DataFrame]
    pandas_result: Optional[dict[str, Any]]
    chart_figure: Any
    chart_type: Optional[str]
    retries: int
    validation_notes: list[str]
    final_answer: str
    profile_text: str


@dataclass
class RunResult:
    """What the Streamlit app consumes after a graph run."""

    answer: str
    tool_used: list[str]
    sql: Optional[str]
    result_df: Optional[pd.DataFrame]
    chart_figure: Any
    notes: list[str]


class DataAnalystAgent:
    """Builds and runs the LangGraph workflow for a single loaded dataset."""

    def __init__(self, df: pd.DataFrame, profile_text: str, gemini: GeminiClient, memory: AgentMemory) -> None:
        self.df = df
        self.profile_text = profile_text
        self.gemini = gemini
        self.memory = memory
        self.graph = self._build_graph()

    # -- graph construction -------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_memory", self._retrieve_memory)
        graph.add_node("understand_question", self._understand_question)
        graph.add_node("decide_tool", self._decide_tool)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("validate_result", self._validate_result)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("save_memory", self._save_memory)

        graph.set_entry_point("retrieve_memory")
        graph.add_edge("retrieve_memory", "understand_question")
        graph.add_edge("understand_question", "decide_tool")
        graph.add_edge("decide_tool", "execute_tool")
        graph.add_edge("execute_tool", "validate_result")
        graph.add_conditional_edges(
            "validate_result",
            self._should_retry,
            {"retry": "execute_tool", "continue": "generate_answer"},
        )
        graph.add_edge("generate_answer", "save_memory")
        graph.add_edge("save_memory", END)
        return graph.compile()

    # -- nodes ----------------------------------------------------------

    def _retrieve_memory(self, state: AgentState) -> AgentState:
        context = self.memory.context_for_prompt(state["question"])
        return {"memory_context": context, "profile_text": self.profile_text}

    def _understand_question(self, state: AgentState) -> AgentState:
        prompt = prompts.UNDERSTAND_QUESTION_PROMPT.format(
            profile=state["profile_text"],
            memory_context=state["memory_context"],
            chat_history=self.memory.conversation.recent_history_text(),
            question=state["question"],
        )
        resolved = self.gemini.generate(prompts.SYSTEM_PROMPT, prompt) or state["question"]
        return {"resolved_question": resolved}

    def _decide_tool(self, state: AgentState) -> AgentState:
        prompt = prompts.TOOL_SELECTION_PROMPT.format(
            profile=state["profile_text"], question=state["resolved_question"]
        )
        raw = self.gemini.generate(prompts.SYSTEM_PROMPT, prompt)
        tools_list = _parse_tool_selection(raw)
        return {"tools_to_use": tools_list, "retries": 0}

    def _execute_tool(self, state: AgentState) -> AgentState:
        result: AgentState = {}

        selected = state.get("tools_to_use", ["sql"])
        question = state["resolved_question"]

        if "profile" in selected:
            profile_result = tools.run_profile_tool(self.df)
            result["pandas_result"] = {
                "profile": profile_result.profile
            }

        if "sql" in selected or _is_diagnostic(question):
            error_context = ""

            if state.get("sql_error"):
                error_context = (
                    f"The previous query failed with error:\n"
                    f"{state['sql_error']}\nFix it."
                )

            sql_prompt = prompts.SQL_GENERATION_PROMPT.format(
                schema=state["profile_text"],
                question=question,
                error_context=error_context,
            )

            sql_text = self.gemini.generate(
                prompts.SYSTEM_PROMPT,
                sql_prompt,
            )

            sql_text = _strip_code_fences(sql_text)

            sql_res = tools.run_sql_tool(
                self.df,
                sql_text,
            )

            result["sql"] = sql_res.sql
            result["sql_result"] = sql_res.result_df
            result["sql_error"] = sql_res.error

        if "pandas" in selected or _is_diagnostic(question):
            result["pandas_result"] = {
                **(result.get("pandas_result") or {}),
                **_run_pandas_analysis(
                    self.df,
                    question,
                ),
            }

        if "chart" in selected:
            sql_df = result.get("sql_result")

            # Never generate a chart from an empty SQL result.
            if sql_df is not None and sql_df.empty:
                result["chart_figure"] = None
                result["chart_type"] = None
            else:
                chart_res = _build_chart_from_context(
                    sql_df=sql_df,
                    pandas_result=result.get("pandas_result"),
                    question=question,
                )

                result["chart_figure"] = chart_res.figure
                result["chart_type"] = chart_res.chart_type

        return result

    def _validate_result(self, state: AgentState) -> AgentState:
        notes: list[str] = []
        if state.get("sql_error"):
            notes.append(f"SQL error: {state['sql_error']}")
        elif "sql" in state.get("tools_to_use", []):
            df_res = state.get("sql_result")
            if df_res is None or df_res.empty:
                notes.append("SQL query returned no rows.")
        return {"validation_notes": notes}

    def _should_retry(self, state: AgentState) -> str:
        if state.get("sql_error") and state.get("retries", 0) < MAX_SQL_RETRIES:
            state["retries"] = state.get("retries", 0) + 1
            return "retry"
        return "continue"

    def _generate_answer(self, state: AgentState) -> AgentState:
        question = state["resolved_question"]

        sql_result = state.get("sql_result")
        selected_tools = state.get("tools_to_use", [])
        validation_notes = state.get("validation_notes", [])

        # ---------------------------------------------------------
        # HARD GUARD:
        # If SQL was used and returned zero rows, do not ask the LLM
        # to invent an answer.
        # ---------------------------------------------------------
        if (
            "sql" in selected_tools
            and sql_result is not None
            and sql_result.empty
            and "pandas" not in selected_tools
        ):
            answer = (
                "No matching records were found for this query."
            )

            return {
                "final_answer": answer
            }

        tool_results_text = _summarize_tool_results(state)

        if (
            _is_diagnostic(question)
            and state.get("pandas_result", {}).get(
                "period_comparison"
            )
        ):
            prompt = prompts.DIAGNOSTIC_BREAKDOWN_PROMPT.format(
                question=question,
                diagnostic_data=json.dumps(
                    state["pandas_result"]["period_comparison"],
                    default=str,
                ),
            )
        else:
            prompt = prompts.FINAL_ANSWER_PROMPT.format(
                question=question,
                tool_results=tool_results_text,
            )

        answer = self.gemini.generate(
            prompts.SYSTEM_PROMPT,
            prompt,
        )

        if validation_notes:
            answer += (
                "\n\n_Note: "
                + "; ".join(validation_notes)
                + "_"
            )

        return {
            "final_answer": answer
        }

    def _save_memory(self, state: AgentState) -> AgentState:
        summary = state["final_answer"][:280]

        # Keep the normal conversation transcript
        self.memory.remember_turn(
            state["question"],
            summary
        )

        # Keep structured analytical context for follow-up questions
        self.memory.remember_analysis_context(
            question=state["question"],
            resolved_question=state.get(
                "resolved_question",
                state["question"]
            ),
            sql=state.get("sql"),
            result_df=state.get("sql_result"),
            tools_used=state.get("tools_to_use", []),
            chart_type=state.get("chart_type"),
        )

        return {}

    # -- public entry point ---------------------------------------------

    def ask(self, question: str) -> RunResult:
        initial_state: AgentState = {"question": question}
        final_state = self.graph.invoke(initial_state)
        return RunResult(
            answer=final_state.get("final_answer", ""),
            tool_used=final_state.get("tools_to_use", []),
            sql=final_state.get("sql"),
            result_df=final_state.get("sql_result"),
            chart_figure=final_state.get("chart_figure"),
            notes=final_state.get("validation_notes", []),
        )


# ---------------------------------------------------------------------------
# Helper functions (pure, no LLM calls)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(sql|json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _parse_tool_selection(raw: str) -> list[str]:
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        chosen = data.get("tools", ["sql"])
        valid = {"sql", "pandas", "chart", "profile"}
        return [t for t in chosen if t in valid] or ["sql"]
    except Exception:  # noqa: BLE001
        return ["sql"]


def _is_diagnostic(question: str) -> bool:
    lowered = question.lower()
    return any(kw in lowered for kw in ["why", "decrease", "decline", "drop", "increase", "spike", "cause"])


def _find_col(df: pd.DataFrame, dtype_kind: str) -> Optional[str]:
    if dtype_kind == "datetime":
        cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    elif dtype_kind == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    else:
        cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cols[0] if cols else None


def _quarter_bounds(year: int, quarter: int) -> tuple[str, str]:
    start_month = (quarter - 1) * 3 + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = (start + pd.offsets.QuarterEnd(0))
    return start.date().isoformat(), end.date().isoformat()


def _run_pandas_analysis(df: pd.DataFrame, question: str) -> dict[str, Any]:
    """Dispatch a simple heuristic pandas analysis based on the question."""
    lowered = question.lower()
    out: dict[str, Any] = {}

    if _is_diagnostic(question):
        date_col = _find_col(df, "datetime")
        value_col = _find_col(df, "numeric")
        breakdown_col = _find_col(df, "categorical")

        if date_col and value_col:
            work = pd.to_datetime(df[date_col], errors="coerce")
            year_match = re.search(r"(20\d{2})", question)
            year = int(year_match.group(1)) if year_match else int(work.dt.year.max())
            quarter_match = re.search(r"[qQ]([1-4])", question)
            quarter = int(quarter_match.group(1)) if quarter_match else int(work.dt.quarter.max())
            prev_quarter, prev_year = (quarter - 1, year) if quarter > 1 else (4, year - 1)

            period_a = _quarter_bounds(year, quarter)
            period_b = _quarter_bounds(prev_year, prev_quarter)

            res = tools.pandas_period_comparison(
                df, date_col, value_col, period_a, period_b, breakdown_col=breakdown_col
            )
            out["period_comparison"] = res.result if not res.error else {"error": res.error}

    if "correlation" in lowered:
        out["correlation"] = tools.pandas_correlation(df).result
    if "missing" in lowered:
        out["missing_analysis"] = tools.pandas_missing_analysis(df).result

    return out


def _build_chart_from_context(
    sql_df: Optional[pd.DataFrame], pandas_result: Optional[dict[str, Any]], question: str
) -> tools.ChartResult:
    if sql_df is not None and not sql_df.empty:
        numeric_cols = sql_df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = [c for c in sql_df.columns if c not in numeric_cols]
        date_col = next((c for c in non_numeric_cols if "date" in c.lower() or "month" in c.lower()), None)
        x_col = date_col or (non_numeric_cols[0] if non_numeric_cols else sql_df.columns[0])
        y_col = numeric_cols[0] if numeric_cols else sql_df.columns[-1]
        color_col = None
        if len(non_numeric_cols) > 1:
            color_col = next((c for c in non_numeric_cols if c != x_col), None)
        chart_type = tools.suggest_chart_type(sql_df, date_col, x_col if not date_col else None, numeric_cols)
        return tools.build_chart(sql_df, chart_type, x=x_col, y=y_col, color=color_col, title=question[:80])

    comparison = (pandas_result or {}).get("period_comparison")
    if comparison and comparison.get("breakdown"):
        breakdown_df = pd.DataFrame.from_dict(comparison["breakdown"], orient="index").reset_index()
        breakdown_df = breakdown_df.rename(columns={"index": "dimension"})
        return tools.build_chart(
            breakdown_df, "bar", x="dimension", y="abs_change", title="Change by dimension"
        )

    return tools.ChartResult(figure=None, chart_type="none", error="No suitable data available for a chart.")


def _summarize_tool_results(state: AgentState) -> str:
    parts: list[str] = []

    if state.get("sql"):
        parts.append(
            f"SQL query used:\n{state['sql']}"
        )

    df_res = state.get("sql_result")

    if df_res is not None:
        if df_res.empty:
            parts.append(
                "SQL result: 0 rows. "
                "No matching records were found."
            )
        else:
            parts.append(
                "SQL result (first rows):\n"
                + df_res.head(20).to_string(index=False)
            )

    if state.get("sql_error"):
        parts.append(
            f"SQL error: {state['sql_error']}"
        )

    if state.get("pandas_result"):
        parts.append(
            "Pandas results:\n"
            + json.dumps(
                state["pandas_result"],
                default=str,
            )[:3000]
        )

    if not parts:
        parts.append(
            "No tool results were produced."
        )

    return "\n\n".join(parts)
