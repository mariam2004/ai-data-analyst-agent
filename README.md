# AI Data Analyst Agent

A small, portfolio-friendly AI agent that lets you upload a CSV/Excel file,
ask questions about it in plain English, and get data-grounded answers,
SQL/Pandas analysis, and Plotly charts — without the LLM ever inventing a
number.

## 1. Overview

Upload a dataset, then chat with it:

- "What are the top 5 products by revenue?"
- "Which region has the highest profit?"
- "Why did sales decrease in Q3?"
- "Create a visualization of monthly sales."
- "Show them by region." (follow-up, resolved via memory)
- "Only for 2025." (refinement, resolved via memory)

The LLM (Gemini) never computes anything itself. It only understands the
question, decides which tool to call, and explains the tool's results.

## 2. Features

- Natural-language Q&A over an uploaded CSV/XLSX/XLS file
- Automatic dataset profiling (rows, columns, dtypes, missing values,
  duplicates, numeric/categorical/datetime columns)
- Read-only analytical SQL via DuckDB, with destructive statements blocked
- Pandas-based analysis for correlation, missing-value analysis, and
  period-over-period diagnostic breakdowns
- Automatic Plotly chart-type selection (line / bar / histogram / scatter)
- Short-term conversation memory for natural follow-up questions
- Lightweight semantic memory (RAG) over conversation facts, using
  `sentence-transformers/all-MiniLM-L6-v2`
- Basic SQL error recovery (up to 2 automatic retries)
- Streamlit chat UI with an expandable "Analysis details" panel showing the
  tool used and the generated SQL
- Generic PDF data analyst report generation with automatic dataset analysis,
  data-quality checks, missing-value breakdowns, and dynamic column-type detection

## 3. Architecture

```text
User
  |
Streamlit (app.py)
  |
LangGraph Agent (agent.py)
  |
retrieve memory --> understand question --> decide tool
                                                 |
                                            execute tool
                                           /     |      \
                                      DuckDB  Pandas   Plotly
                                           \     |      /
                                            validate result
                                                 |
                                          Gemini (explain)
                                                 |
                                            save memory
                                                 |
                                            Final Answer
```

Strict separation of concerns:

```text
LLM (Gemini)  -> understands, plans, chooses tools, explains results
Tools         -> DuckDB / Pandas / Plotly actually calculate and render
Memory        -> short-term conversation transcript
RAG           -> retrieves relevant conversation memory for follow-ups
```

The dataset itself is always the source of truth for any number that
appears in an answer.

## 4. Tech stack

- **UI:** Streamlit
- **LLM:** Google Gemini (`google-genai` SDK), model configurable in one
  place (`agent.py` -> `DEFAULT_MODEL`)
- **Agent orchestration:** LangGraph + LangChain
- **SQL analytics:** DuckDB (in-memory, queries a Pandas DataFrame directly)
- **Data wrangling:** Pandas
- **Visualization:** Plotly
- **Reporting:** ReportLab + Matplotlib
- **PDF analysis:** Automatic numeric, categorical, datetime, metadata, and identifier detection
- **Conversation memory:** in-process transcript
- **Semantic memory / RAG:** `sentence-transformers` embeddings, small
  in-memory vector store (no external vector DB)

## 5. Project structure

```text
ai-data-analyst-agent/
├── app.py            # Streamlit UI
├── agent.py          # LangGraph workflow + Gemini client
├── tools.py          # DuckDB / Pandas / Plotly tools
├── data_utils.py      # File loading, validation, profiling
├── memory.py         # Conversation memory + lightweight RAG
├── prompts.py        # All LLM prompts
├── report.py         # Generic PDF data analyst report generation
├── requirements.txt
├── .env               # GEMINI_API_KEY goes here
├── data/              # Place your own sample CSV/Excel file here
└── README.md
```

## 6. Installation

```bash
git clone <this-repo>
cd ai-data-analyst-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 7. Environment variable setup

Edit the `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
The key is loaded via `python-dotenv` and is never printed or logged.

## 8. How to run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Place any dataset you'd like to test with inside `data/` (this repo does
not ship a sample file — bring your own CSV or Excel export), then upload it
through the UI.

## 9. Example questions

- "What are the top 5 products by revenue?"
- "Which region has the highest profit?"
- "How many rows have missing revenue?"
- "Create a visualization of monthly sales."
- "Why did sales decrease in Q3?"
- "Show them by region."
- "Only for 2025."

## 10. Datasets Used

The project was tested with publicly available tabular datasets from
Tableau Public's official Sample Data collection.

### Tableau Superstore Sales

A fictional retail dataset containing product, sales, profit, quantity,
discount, customer, and regional information.

🔗 [Tableau Public — Sample Data](https://public.tableau.com/app/resources/sample-data)

### The 2014 Inc. 5000

A dataset containing information about the 5,000 fastest-growing private
companies listed in the 2014 Inc. 5000, including revenue, growth,
employees, industry, and location.

🔗 [Tableau Public — Sample Data](https://public.tableau.com/app/resources/sample-data)

## 11. How memory works

Two lightweight layers work together so follow-up questions feel natural:

1. **Short-term conversation memory** — the recent chat transcript is kept
   in `AgentMemory.conversation` and fed into the "understand question" step
   so the agent can resolve pronouns and references ("them", "same but...").
2. **Lightweight semantic memory (RAG)** — after each turn, a short factual
   note (e.g. "Q: top 5 products by revenue -> A, B, C, D, E") is embedded
   with `all-MiniLM-L6-v2` and stored in a small in-memory vector store.
   When a new question comes in, the most semantically relevant notes are
   retrieved and added to the prompt as context.

RAG here is **only** used for conversation/context memory — it never
analyzes the dataset itself. All numeric analysis always goes through
DuckDB or Pandas.

Memory is in-process and per-session; it does not persist across
application restarts in this version.

## 12. Generated Reports

Example PDF reports generated by the application are included in the

eports/ directory.

- 📄 [Superstore Data Analyst Report](reports/AI_Data_Analyst_Report_Superstore.pdf)
- 📄 [Inc. 5000 Data Analyst Report](reports/AI_Data_Analyst_Report_Inc5000.pdf)

These reports demonstrate the generic PDF reporting capability of the
application across different dataset structures.

## 13. Security / limitations

- Only single, read-only `SELECT`/`WITH` SQL statements are allowed.
  Destructive keywords (`DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`,
  `CREATE`, `ATTACH`, `COPY`, etc.) are blocked before execution.
- The LLM never runs arbitrary generated Python — Pandas operations are
  limited to a small, controlled set of functions in `tools.py`.
- No authentication, persistence, or multi-user isolation is implemented;
  this is a single-session demo/portfolio app, not a production system.
- Diagnostic ("why") analysis is heuristic (period comparison + breakdown
  by one dimension) and explicitly avoids claiming causation the data
  doesn't support.
- Large files may be slow to profile/embed since everything runs in-process
  with no background workers.
- PDF reports are generated dynamically from the uploaded dataset and use
  automatic schema detection rather than a fixed business-specific schema.

## 14. Future improvements

- Persist conversation and semantic memory across sessions
- Support multiple uploaded datasets and joins between them
- Add authentication and multi-user session isolation
- Cache repeated SQL/Pandas computations
- Add more chart types and user-driven chart customization
- Stream the LLM's final answer token-by-token in the UI
