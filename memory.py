
"""
Memory layers for the AI Data Analyst Agent.

1. Short-term conversation memory:
   Stores the chat transcript for the current session.

2. Semantic memory (lightweight RAG):
   Stores short factual notes using embeddings.

3. Structured analysis memory:
   Stores the latest analytical result in a structured form so
   follow-up references like "them", "those products", or
   "only for the West region" can be resolved correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json

import numpy as np


@dataclass
class ConversationMemory:
    """Short-term, per-session chat transcript."""

    turns: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})

    def recent_history_text(self, max_turns: int = 6) -> str:
        recent = self.turns[-max_turns:]
        if not recent:
            return "(no previous conversation)"

        lines = [f"{t['role']}: {t['content']}" for t in recent]
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns.clear()


class SemanticMemory:
    """Lightweight semantic memory using sentence-transformers."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model = None
        self._notes: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def add_note(self, note: str) -> None:
        if not note or not note.strip():
            return

        try:
            model = self._get_model()
            embedding = model.encode(
                note,
                normalize_embeddings=True
            )
        except Exception:
            return

        self._notes.append(note.strip())
        self._embeddings.append(np.asarray(embedding))

    def search(self, query: str, top_k: int = 3) -> list[str]:
        if not self._notes:
            return []

        try:
            model = self._get_model()
            query_vec = np.asarray(
                model.encode(
                    query,
                    normalize_embeddings=True
                )
            )
        except Exception:
            return self._notes[-top_k:]

        scores = [
            float(np.dot(query_vec, vec))
            for vec in self._embeddings
        ]

        ranked = sorted(
            zip(scores, self._notes),
            key=lambda x: x[0],
            reverse=True
        )

        return [note for _, note in ranked[:top_k]]

    def clear(self) -> None:
        self._notes.clear()
        self._embeddings.clear()


@dataclass
class AgentMemory:
    """Combines short-term, semantic, and structured analysis memory."""

    conversation: ConversationMemory = field(
        default_factory=ConversationMemory
    )

    semantic: SemanticMemory = field(
        default_factory=SemanticMemory
    )

    latest_analysis_context: dict[str, Any] = field(
        default_factory=dict
    )

    def context_for_prompt(
        self,
        question: str,
        top_k: int = 3
    ) -> str:
        """Build the memory context used by the agent."""

        blocks: list[str] = []

        # ---------------------------------------------------------
        # 1. Structured context from the most recent analysis
        # ---------------------------------------------------------
        if self.latest_analysis_context:
            structured_text = json.dumps(
                self.latest_analysis_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            blocks.append(
                "CURRENT ANALYSIS CONTEXT:\n"
                "Use this as the primary source when resolving "
                "follow-up references such as "
                "\"them\", \"these\", \"those\", or \"the previous results\".\n"
                "Do not recompute a previous Top-N selection unless "
                "the user explicitly asks for a new ranking.\n\n"
                f"{structured_text}"
            )

        # ---------------------------------------------------------
        # 2. Semantic memory
        # ---------------------------------------------------------
        relevant = self.semantic.search(
            question,
            top_k=top_k
        )

        if relevant:
            semantic_text = "\n".join(
                f"- {note}"
                for note in relevant
            )

            blocks.append(
                "RELEVANT SEMANTIC MEMORY:\n"
                f"{semantic_text}"
            )

        if not blocks:
            return "(no relevant stored context)"

        return "\n\n".join(blocks)

    def remember_turn(
        self,
        question: str,
        answer_summary: str
    ) -> None:
        """Store the human-readable conversation turn."""

        self.conversation.add_user_message(question)
        self.conversation.add_assistant_message(answer_summary)

        self.semantic.add_note(
            f"Q: {question} -> {answer_summary}"
        )

    def remember_analysis_context(
        self,
        question: str,
        resolved_question: str,
        sql: str | None,
        result_df: Any = None,
        tools_used: list[str] | None = None,
        chart_type: str | None = None,
        max_rows: int = 10,
    ) -> None:
        """
        Store the latest analysis in structured form.

        Only a small result sample is stored.
        The original dataset is NEVER stored in memory.
        """

        context: dict[str, Any] = {
            "question": question,
            "resolved_question": resolved_question,
            "sql": sql,
            "tools_used": tools_used or [],
            "chart_type": chart_type,
        }

        if result_df is not None:
            try:
                columns = [
                    str(col)
                    for col in result_df.columns
                ]

                rows = result_df.head(max_rows).to_dict(
                    orient="records"
                )

                context["result_columns"] = columns
                context["result_rows"] = rows
                context["result_row_count"] = int(
                    len(result_df)
                )

                # Store non-numeric columns separately.
                entity_columns: dict[str, list[Any]] = {}

                for col in columns:
                    try:
                        values = (
                            result_df[col]
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                            .head(10)
                            .tolist()
                        )

                        if values:
                            entity_columns[col] = values
                    except Exception:
                        continue

                context["entities"] = entity_columns

            except Exception:
                pass

        self.latest_analysis_context = context

        # Also keep a compact semantic note.
        if result_df is not None:
            try:
                note = (
                    f"Previous analysis: {question}. "
                    f"SQL result columns: "
                    f"{list(result_df.columns)}. "
                    f"Key result rows: "
                    f"{result_df.head(5).to_dict(orient='records')}"
                )
                self.semantic.add_note(note)
            except Exception:
                pass

    def remember_fact(self, fact: str) -> None:
        """Persist a useful factual note."""
        self.semantic.add_note(fact)

    def clear(self) -> None:
        """Clear all memory layers."""
        self.conversation.clear()
        self.semantic.clear()
        self.latest_analysis_context.clear()
