# AI Data Analyst Agent

A Streamlit app that lets you upload a CSV or Excel file, ask questions about it in plain English, and get answers grounded in tool output. Google Gemini interprets the question, picks tools and writes the explanation; DuckDB, Pandas and Plotly do the actual calculating and charting. The app can also generate a rule-based PDF report of the dataset.

## Table of contents

1. [Features](#1-features)
2. [Architecture and workflow](#2-architecture-and-workflow)
3. [AI / LLM integration](#3-ai--llm-integration)
4. [Tools](#4-tools)
5. [Memory](#5-memory)
6. [Streamlit application](#6-streamlit-application)
7. [PDF report](#7-pdf-report)
8. [Supported data formats](#8-supported-data-formats)
9. [Project structure](#9-project-structure)
10. [Technologies](#10-technologies)
11. [Installation](#11-installation)
12. [Configuration (`.env`)](#12-configuration-env)
13. [Running the app](#13-running-the-app)
14. [Usage](#14-usage)
15. [Limitations and requirements](#15-limitations-and-requirements)

---

## 1. Features

- Chat with one uploaded dataset in natural language.
- Automatic dataset profile: row and column counts, dtypes, missing values, duplicate rows, numeric / categorical / datetime column lists, and basic numeric statistics.
- Read-only SQL analytics with DuckDB. The uploaded DataFrame is exposed as a table named `data`.
- Pandas analysis for period-over-period diagnostic ("why") questions, correlation and missing-value analysis.
- Plotly charts (line, bar, scatter) chosen automatically from the shape of the result.
- Follow-up questions ("show them by region", "only for 2025") resolved through session memory.
- Retry of failed SQL through a conditional edge in the workflow (see [limitations](#15-limitations-and-requirements) for a caveat).
- "Analysis details" panel under each answer showing the tools used, the SQL that ran, and validation notes.
- One-click PDF data report with tables and charts, generated without any LLM call.

## 2. Architecture and workflow

```text
User
  |
Streamlit UI (app.py)
  |
DataAnalystAgent (agent.py, LangGraph StateGraph)
  |
retrieve_memory -> understand_question -> decide_tool -> execute_tool
                                                             |
                                            +----------------+----------------+
                                            |                |                |
                                        DuckDB SQL        Pandas           Plotly
                                            |                |                |
                                            +----------------+----------------+
                                                             |
                                                      validate_result
                                                   /                 \
                                         (SQL error, retry)        (continue)
                                                 |                     |
                                           execute_tool         generate_answer
                                                                       |
                                                                  save_memory
                                                                       |
                                                                     END
```

Node by node (`agent.py`):

| Node | What it does |
|---|---|
| `retrieve_memory` | Builds the memory context for the question (structured last-analysis context plus semantic notes) and attaches the dataset profile text. |
| `understand_question` | Asks Gemini to rewrite the question as one self-contained request, resolving references such as "them" using memory and the last 6 chat messages. |
| `decide_tool` | Asks Gemini for a JSON list of tools. Valid values: `sql`, `pandas`, `chart`, `profile`. Falls back to `["sql"]` if the reply is unparseable or contains no valid tool. |
| `execute_tool` | Runs the selected tools (details in [Tools](#4-tools)). |
| `validate_result` | Adds notes when SQL errored or returned no rows. |
| conditional edge `_should_retry` | Returns to `execute_tool` while a SQL error exists and the retry counter is below `MAX_SQL_RETRIES = 2`; otherwise continues. |
| `generate_answer` | Returns a fixed "No matching records were found for this query." message when SQL returned zero rows and Pandas was not selected. Otherwise asks Gemini to write the answer from tool results only. |
| `save_memory` | Stores the turn and the structured analysis context. |

### Where LangGraph is used, and why

LangGraph is imported only in `agent.py` (`from langgraph.graph import END, StateGraph`). It provides the state machine: `DataAnalystAgent._build_graph()` registers the seven nodes above on a `StateGraph(AgentState)`, wires the edges, adds the conditional retry edge, and compiles the graph. `DataAnalystAgent.ask()` calls `graph.invoke(...)` once per user question. The shared state type is `AgentState` (a `TypedDict`).

## 3. AI / LLM integration

- **Provider / SDK:** Google Gemini through the `google-genai` SDK (`from google import genai`).
- **Model:** set in one place, `DEFAULT_MODEL` in `agent.py` (currently `"gemini-3.5-flash-lite"`). Change that constant to switch models.
- **Client:** `GeminiClient.generate(system_instruction, user_prompt)` wraps `client.models.generate_content` with `temperature = 0.2`. Transient API errors (HTTP 408, 429, 500, 502, 503, 504) are retried up to 3 attempts in total, with exponential backoff and jitter. Other errors are raised.
- **Prompts:** all in `prompts.py`:
  - `SYSTEM_PROMPT` — rules: never calculate numbers, never invent results, read-only SQL only.
  - `UNDERSTAND_QUESTION_PROMPT` — follow-up resolution.
  - `TOOL_SELECTION_PROMPT` — tool choice as JSON.
  - `SQL_GENERATION_PROMPT` — one DuckDB `SELECT` against table `data`.
  - `FINAL_ANSWER_PROMPT` — answer strictly from tool results.
  - `DIAGNOSTIC_BREAKDOWN_PROMPT` — summary for "why" questions using computed period comparisons.
- **What is sent to Gemini:** the dataset profile text (schema, dtypes, missing counts, duplicates, column groups), the question, memory context, and tool results (SQL text, the first 20 rows of the SQL result, and Pandas results truncated to 3,000 characters). The full DataFrame is never sent, but aggregated result rows and stored result samples can be.
- The PDF report (`report.py`) makes no LLM calls.

## 4. Tools

Implemented in `tools.py` and dispatched from `DataAnalystAgent._execute_tool` in `agent.py`.

**Profile tool** (`run_profile_tool`) — runs `profile_data` from `data_utils.py`. Used when the LLM selects `profile`.

**SQL tool** (`run_sql_tool`, DuckDB) — runs in an in-memory DuckDB connection with the DataFrame registered as `data`. `validate_sql` only accepts a single statement that starts with `SELECT` or `WITH`, and rejects queries containing any of these keywords: `drop, delete, update, insert, alter, create, attach, detach, copy, pragma, export, import, install, load, call, vacuum, replace, grant, revoke`. The SQL is generated by Gemini (code fences are stripped). It runs when `sql` is selected, and also for any question the code treats as diagnostic.

**Pandas tools** — `agent._run_pandas_analysis` uses:

- `pandas_period_comparison`: for diagnostic questions (containing words such as *why, decrease, decline, drop, increase, spike, cause*), compares a metric between a quarter and the previous quarter, with a breakdown by one dimension. The year and quarter are parsed from the question (`20xx`, `Qn`), otherwise the latest ones in the data are used. It uses the first datetime-typed column, the first numeric column and the first text/category column.
- `pandas_correlation`: when the question contains "correlation".
- `pandas_missing_analysis`: when the question contains "missing".

`pandas_describe` exists in `tools.py` but is not called by the agent.

**Chart tool** (`build_chart`, `suggest_chart_type`, Plotly Express) — runs when `chart` is selected. It charts the SQL result (x = first date/month-named column, else first non-numeric column; y = first numeric column; optional colour = second non-numeric column). If there is no SQL result, it plots the "change by dimension" bar chart from a period comparison. It never charts an empty SQL result. Automatic selection only picks `line`, `bar` or `scatter`; `build_chart` also supports `histogram` and `stacked_bar`, but the agent never chooses them.

## 5. Memory

Implemented in `memory.py` and held in the Streamlit session; nothing is written to disk.

1. **Conversation memory** — the transcript. The last 6 messages are passed to `understand_question`. Assistant answers are stored truncated to 280 characters.
2. **Structured analysis memory** — the latest analysis only: question, resolved question, SQL, tools used, chart type, up to 10 result rows, result columns and distinct entity values. This is what lets "them" refer to the previous top-N result.
3. **Semantic memory** — short notes embedded with `sentence-transformers/all-MiniLM-L6-v2` and kept in an in-memory list. The 3 most similar notes to a new question are added to the prompt. The model loads lazily on first use. If it cannot be loaded, notes are silently skipped and the agent continues without semantic recall.

Memory is reset whenever a different file is uploaded.

## 6. Streamlit application

`app.py` is the entry point.

1. Loads `.env` with `python-dotenv` and reads `GEMINI_API_KEY`. If missing, an error banner is shown and chat is disabled.
2. File uploader (`csv`, `xlsx`, `xls`). On upload, the file is loaded, validated and profiled. If the dataset differs from the one in session state, chat history and memory are reset and a new `DataAnalystAgent` is built (only when an API key exists).
3. "Dataset overview" expander: rows, columns, duplicate rows, column dtypes, missing values.
4. Chat: each answer can include the result table, a Plotly chart, and an "Analysis details" expander (tools used, SQL, notes). Errors from the agent are shown in the chat instead of crashing the app.
5. "Data Analyst Report" section with a Generate button and a PDF download button (see below).

## 7. PDF report

`report.py` → `generate_pdf_report(df, dataset_name)` builds an A4 PDF with ReportLab and Matplotlib charts, using rule-based column classification (numeric, categorical, datetime, identifier and metadata columns are detected from names and values). Sections: Executive Summary, Dataset Overview, Key Numeric Metrics, Categorical Analysis, Time-Based Analysis, Category vs Numeric Analysis, Numeric Relationship, Data Quality, Detected Column Types. Sections that lack suitable columns are skipped. Limits in the code include the first 10 numeric columns, the first 4 categorical columns, top-10 category tables, and the 15 columns with the most missing values. Scatter plots use at most 2,000 rows. The download is named `AI_Data_Analyst_Report.pdf`.

## 8. Supported data formats

Defined in `data_utils.load_file` and the uploader in `app.py`:

| Extension | Reader | Notes |
|---|---|---|
| `.csv` | `pandas.read_csv` | Default pandas parsing (comma delimiter, default encoding). Errors for empty or malformed files are reported. |
| `.xlsx` | `pandas.read_excel` | Requires `openpyxl`. First sheet only. |
| `.xls` | `pandas.read_excel` | Requires `xlrd`. First sheet only. |

Any other extension is rejected. Files with no rows or no columns are rejected.

## 9. Project structure

```text
ai-data-analyst-agent/
├── app.py             # Streamlit UI: upload, overview, chat, PDF report
├── agent.py           # GeminiClient + LangGraph workflow (DataAnalystAgent)
├── tools.py           # SQL validation/execution (DuckDB), Pandas analyses, Plotly charts
├── data_utils.py      # File loading, validation, dataset profiling
├── memory.py          # Conversation, structured, and semantic (embedding) memory
├── prompts.py         # All LLM prompts
├── report.py          # PDF report generation (ReportLab + Matplotlib)
├── requirements.txt   # Python dependencies
├── .env.example       # Template for GEMINI_API_KEY
└── data/
    └── .gitkeep       # Empty folder for your own datasets (no sample data included)
```

The uploaded archive contains exactly this one flat set of files. No nested copies, old duplicates or extra packages were found, so every `.py` file above belongs to the running application. The app does not read `data/`; you upload files through the UI.

## 10. Technologies

| Purpose | Library |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini via `google-genai` |
| Agent workflow | LangGraph |
| SQL analytics | DuckDB |
| Data handling | Pandas, NumPy, openpyxl, xlrd |
| Charts | Plotly (chat), Matplotlib (PDF) |
| PDF | ReportLab |
| Semantic memory | sentence-transformers |
| Configuration | python-dotenv |

## 11. Installation

Requirements: Python 3.10 or newer (developed and smoke-tested on Python 3.12) and a Gemini API key.

```bash
git clone <your-repository-url>
cd ai-data-analyst-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch, so the first install is large.

## 12. Configuration (`.env`)

Copy the template and add your own key:

```bash
cp .env.example .env             # Windows: copy .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey). Never commit `.env`; add it to your `.gitignore` (the archive does not include a `.gitignore`).

## 13. Running the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 14. Usage

1. Upload a `.csv`, `.xlsx` or `.xls` file.
2. Review the "Dataset overview".
3. Ask a question in the chat box, for example:
   - "What are the top 5 products by revenue?"
   - "Show them by region." (follow-up)
   - "Create a visualization of monthly sales."
   - "How many rows have missing revenue?"
   - "Why did sales decrease in Q3?"
4. Open "Analysis details" under an answer to see the tools used and the SQL.
5. To get the PDF, scroll to "Data Analyst Report", click **Generate PDF Report**, then **Download PDF Report**.

The first question in a session may be slower while the embedding model loads.

