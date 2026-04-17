"""
The RAG verification loop and final answer generation.

This pattern is sometimes called "self-RAG" or "adaptive retrieval".
It catches cases like:
  - Synonyms: user asks about "ML models", docs use "neural networks"
  - Decomposition: a compound question needs two separate retrievals
  - Specificity: broad query returns generic chunks; narrower query works better
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache

import anthropic
from dotenv import load_dotenv

from app.embedder import query_collection
from app.models import AskResponse, RetrievalAssessment, RetrievedContext

load_dotenv()
logger = logging.getLogger(__name__)

# config
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
MAX_VERIFICATION_LOOPS = int(os.getenv("MAX_VERIFICATION_LOOPS", "3"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_CACHE_SIZE = 1

# async anthropic client
# using lru_cache to cache the client
@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_anthropic_client() -> anthropic.AsyncAnthropic:
    """Return (and cache) the async Anthropic client."""
    return anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# tool definition for structured assessment of retrieval quality
ASSESSMENT_TOOL: dict = {
    "name": "assess_retrieval",
    "description": (
        "Assess whether the retrieved document chunks contain sufficient information "
        "to accurately answer the user's question. You MUST call this tool."
    ),
    "input_schema": RetrievalAssessment.model_json_schema(),
}


# format chunks for prompt

def _format_context(chunks: list[RetrievedContext]) -> str:
    """
    Render a list of chunks into a readable string for inclusion in a prompt.

    Each chunk is numbered and prefixed with its source file so Claude
    can reference it in reasoning (e.g. "chunk 2 from report.txt covers X").
    """
    if not chunks:
        return "No relevant context was retrieved."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | source: {chunk.source} | distance: {chunk.distance:.3f}]\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


# assess whether retrieved context is sufficient

async def _assess_context(
    question: str,
    chunks: list[RetrievedContext],
    iteration: int,
) -> RetrievalAssessment:
    """
    Ask Claude to evaluate the retrieved chunks against the question.

    Uses tool use so the response is guaranteed to be a valid
    RetrievalAssessment — no text parsing needed.

    Parameters
    ----------
    question  : The original user question.
    chunks    : The chunks retrieved from ChromaDB this iteration.
    iteration : Current loop count (included in prompt for context).
    """
    client = get_anthropic_client()
    context_str = _format_context(chunks)

    system_prompt = (
        "You are a retrieval quality assessor for a RAG system. "
        "Your job is to judge whether the provided document chunks contain "
        "enough information to answer the user's question accurately and completely. "
        "Be honest and critical — partial or tangentially related context should "
        "lower your confidence score. You MUST call the assess_retrieval tool."
    )

    user_message = (
        f"QUESTION (iteration {iteration} of {MAX_VERIFICATION_LOOPS}):\n"
        f"{question}\n\n"
        f"RETRIEVED CONTEXT:\n"
        f"{context_str}\n\n"
        "Assess whether this context is sufficient to answer the question. "
        "If it is not sufficient, propose a new_search_term that is different "
        "from the original question and likely to retrieve better chunks."
    )

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=512,
        system=system_prompt,
        tools=[ASSESSMENT_TOOL],
        # "any" so Claude MUST call at least one of the provided tools to return a RetrievalAssessment
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": user_message}],
    )

    # extracting tool_use from the response to retrieve the RetrievalAssessment
    tool_use_block = next(
        block for block in response.content if block.type == "tool_use"
    )

    # extra validation of the RetrievalAssessment
    assessment = RetrievalAssessment.model_validate(tool_use_block.input)

    logger.info(
        "Iteration %d assessment: confidence=%.2f needs_more_data=%s",
        iteration,
        assessment.confidence_score,
        assessment.needs_more_data,
    )
    return assessment


# generate final answer

async def _generate_answer(
    question: str,
    chunks: list[RetrievedContext],
    assessment: RetrievalAssessment,
) -> str:
    """
    Generate a grounded final answer from the best context available.

    This is a standard (non-tool-use) Claude call. The system prompt
    explicitly tells Claude to stay within the provided context and to
    acknowledge uncertainty if the context is incomplete.
    """
    client = get_anthropic_client()
    context_str = _format_context(chunks)

    confidence_note = (
        f"Note: the retrieval confidence for this context was "
        f"{assessment.confidence_score:.0%}. "
        + ("The context may be incomplete." if assessment.needs_more_data else "")
    )

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided document context. "
        "If the context does not contain enough information to answer fully, "
        "say so explicitly rather than guessing. "
        "Cite which chunks support your answer where relevant."
    )

    user_message = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"{confidence_note}\n\n"
        "Please provide a thorough answer based on the context above."
    )

    response = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# public entry point

async def run_rag_pipeline(question: str) -> AskResponse:
    """
    Run the full RAG pipeline with the verification loop.

    This is the only function called from the FastAPI endpoint.

    Flow
    ────
    for iteration in 1..MAX_VERIFICATION_LOOPS:
        1. Query ChromaDB with the current search term
        2. Ask Claude to assess context quality  →  RetrievalAssessment
        3. If needs_more_data=False  →  break and answer
           If needs_more_data=True and iterations remain  →  use new_search_term
           If needs_more_data=True but no iterations left →  answer anyway

    Then: generate final answer from the best context found.

    Parameters
    ----------
    question : The user's natural-language question.

    Returns
    -------
    AskResponse with the answer, confidence score, sources, and iteration count.
    """
    search_term = question
    best_chunks: list[RetrievedContext] = []
    best_assessment: RetrievalAssessment | None = None
    final_iteration = 1

    for iteration in range(1, MAX_VERIFICATION_LOOPS + 1):
        final_iteration = iteration
        logger.info("Verification loop iteration %d | query: %r", iteration, search_term)

        # retrieve, (blocks the event loop, but it's a fast I/O operation)
        # can be rememdied with asyncio.to_thread 
        # or running chromaDB as a separate server (AsyncHttpClient)
        chunks = query_collection(search_term, n_results=TOP_K_RESULTS)

        if not chunks:
            logger.warning("ChromaDB returned no chunks — is the store empty?")
            best_chunks = []
            break

        assessment = await _assess_context(question, chunks, iteration)

        best_chunks = chunks
        best_assessment = assessment

        if not assessment.needs_more_data:
            logger.info("Context sufficient at iteration %d — proceeding to answer.", iteration)
            break

        # claude wants to re-query
        if iteration < MAX_VERIFICATION_LOOPS:
            logger.info(
                "Needs more data. Re-querying with: %r", assessment.new_search_term
            )
            search_term = assessment.new_search_term
        else:
            # return answer when loop cap is hit
            logger.warning(
                "Reached MAX_VERIFICATION_LOOPS (%d). Answering with best available context.",
                MAX_VERIFICATION_LOOPS,
            )

    # placeholder if db is empty and we have no initial assessment
    if best_assessment is None:
        best_assessment = RetrievalAssessment(
            confidence_score=0.0,
            needs_more_data=False,
            reasoning="No documents found in the store.",
        )

    # generate the final answer
    answer = await _generate_answer(question, best_chunks, best_assessment)

    # de-deuplicate sources
    sources = sorted({chunk.source for chunk in best_chunks})

    return AskResponse(
        answer=answer,
        confidence_score=best_assessment.confidence_score,
        sources=sources,
        iterations=final_iteration,
    )
