"""
FastAPI application — the HTTP layer of the RAG pipeline.

"""

from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.embedder import collection_count, get_collection, get_embedding_function
from app.models import AskRequest, AskResponse
from app.rag_chain import get_anthropic_client, run_rag_pipeline

load_dotenv()

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

# log format
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
 # terminal handler
_console = logging.StreamHandler()
_console.setFormatter(_fmt)

# file handler  
_file = RotatingFileHandler(
    _LOG_DIR / "app.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file.setFormatter(_fmt)

# set the root logger level and add the handlers
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_console)
logging.root.addHandler(_file)

logger = logging.getLogger(__name__)


# lifespan context manager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once when the server starts, and once when it shuts down.

    """
    logger.info("Starting up — warming up embedding model and ChromaDB…")

    # loading and caching the embedding function, chroma client and anthropic client
    get_embedding_function()
    get_collection()
    get_anthropic_client()

    chunk_count = collection_count()
    logger.info("ChromaDB ready — %d chunks in store.", chunk_count)

    if chunk_count == 0:
        logger.warning(
            "The document store is empty. Run `python ingest.py` first, "
            "otherwise /ask will return answers with no context."
        )

    yield  # ← server is running; handle requests

    logger.info("Shutting down.")


# app factory

app = FastAPI(
    title="RAG Pipeline",
    description=(
        "A Retrieval-Augmented Generation API backed by ChromaDB and Claude 3.5 Sonnet. "
        "Features a self-auditing verification loop that re-queries the document store "
        "if Claude judges the initial context insufficient."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# middleware to log request timing

@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code, elapsed)
    return response


# exception handler

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred.", "type": type(exc).__name__},
    )


# routes

@app.get("/health", summary="Health check", tags=["ops"])
async def health() -> dict:
    """
    Returns the current status of the service and the document store size.

    """
    return {
        "status": "ok",
        "model": os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
        "chunk_count": collection_count(),
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question against the document store",
    tags=["rag"],
)
async def ask(request: AskRequest) -> AskResponse:
    """
    Run the RAG verification loop and return a grounded answer.

    **Request body**
    ```json
    { "question": "What are the main risks of transformer architectures?" }
    ```

    **Response**
    ```json
    {
        "answer": "Based on the retrieved context...",
        "confidence_score": 0.87,
        "sources": ["transformers_paper.txt"],
        "iterations": 1
    }
    ```
    """
    if collection_count() == 0:
        raise HTTPException(
            status_code=503, #service unavailable
            detail=(
                "The document store is empty. "
                "Add files to /data and run `python ingest.py` first."
            ),
        )

    # no try/except here as the global exception handler catches anything that escapes
    return await run_rag_pipeline(request.question)
