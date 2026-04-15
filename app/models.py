"""
Pydantic models for the RAG pipeline.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class RetrievalAssessment(BaseModel):
    """
    Claude's structured verdict on whether the retrieved chunks are
    sufficient to answer the user's question.

    Fields
    ------
    confidence_score : float
        How confident Claude is that the retrieved context contains
        enough information to answer the question accurately.
        0.0 = no relevant context at all
        1.0 = context fully and directly answers the question

    needs_more_data : bool
        True  → the context is incomplete/irrelevant; Claude wants to
                 re-query ChromaDB with a different search term.
        False → the context is sufficient; proceed to final answer.

    reasoning : str
        Claude explains *why* it gave this score and this flag.
        Forces the model to reason before deciding, which improves
        accuracy (similar to chain-of-thought prompting).

    new_search_term : str | None
        Only populated when needs_more_data=True.
        Claude proposes a *different* query to search ChromaDB with —
        e.g. a rephrased question, a related concept, or a sub-topic.
        None when needs_more_data=False (no re-query needed).
    """

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that retrieved context answers the question (0–1)",
    )
    needs_more_data: bool = Field(
        ...,
        description="True if the context is insufficient and a re-query is needed",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Claude's explanation of its confidence assessment",
    )
    new_search_term: Optional[str] = Field(
        default=None,
        description="A new ChromaDB query term; only set when needs_more_data=True",
    )

    # enforce that new_search_term is needed when needs_more_data is True
    @model_validator(mode="after")
    def search_term_required_when_needs_more_data(self) -> "RetrievalAssessment":
        if self.needs_more_data and not self.new_search_term:
            raise ValueError(
                "new_search_term must be provided when needs_more_data is True"
            )
        return self


class AskRequest(BaseModel):
    """
    The JSON body the caller sends to POST /ask.

    Example
    -------
    {
        "question": "What are the main risks of transformer architectures?"
    }
    """

    question: str = Field(
        ...,
        min_length=3,
        description="The natural-language question to answer from the document store",
    )

    # strip whitespace from the question
    @field_validator("question", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class AskResponse(BaseModel):
    """
    The JSON body returned by POST /ask.

    Fields
    ------
    answer : str
        Claude's final answer, grounded in the retrieved document chunks.

    confidence_score : float
        The confidence_score from the *last* RetrievalAssessment before
        Claude generated the final answer. Lets callers know how sure
        the system was about the context quality.

    sources : list[str]
        The source file names of the chunks used in the final answer.
        Useful for the caller to trace back which documents were cited.

    iterations : int
        How many verification loop cycles ran before an answer was given.
        iterations=1 means the first retrieval was good enough.
        iterations=3 means Claude re-queried twice before answering.
        Useful for debugging retrieval quality.
    """

    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[str]
    iterations: int = Field(ge=1)


# internal context carrier
# used internally by rag_chain.py to pass retrieved chunks 
# between the verification loop and answer generation.

class RetrievedContext(BaseModel):
    """
    A single document chunk returned by ChromaDB.

    ChromaDB returns parallel lists (documents, metadatas, distances)
    so we zip them into these typed objects for cleaner code downstream.
    """

    text: str = Field(..., description="The raw chunk text")
    source: str = Field(..., description="Filename the chunk came from")
    chunk_index: int = Field(..., description="Position of this chunk in its source file")
    distance: float = Field(..., description="Embedding distance (lower = more similar)")
