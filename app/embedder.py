"""
ChromaDB client, embedding function, and query helper.
"""

import os
from functools import lru_cache

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

from app.models import RetrievedContext

load_dotenv()

# config

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", ".chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# embedding model: to switch, delete .chroma/ and re-run ingest.py.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# max size of the cache
MAX_CACHE_SIZE = 1

# singleton helpers
# using lru_cache to cache the embedding function and chroma client

@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Return (and cache) the SentenceTransformer embedding function."""
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return (and cache) a ChromaDB PersistentClient.

    PersistentClient saves data to disk at CHROMA_PERSIST_DIR so your
    embedded documents survive server restarts. Use chromadb.Client()
    instead if you want an in-memory-only store (useful for testing).
    """
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_collection() -> chromadb.Collection:
    """
    Return the ChromaDB collection, creating it if it doesn't exist yet.

    get_or_create_collection is idempotent — safe to call on every startup.
    We pass the embedding function here so ChromaDB knows how to embed
    any text we add or query with later.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# core operations

def add_chunks(
    texts: list[str],
    ids: list[str],
    metadatas: list[dict],
) -> None:
    """
    Embed and store document chunks in ChromaDB.

    Parameters
    ----------
    texts     : The raw text of each chunk.
    ids       : Unique string ID for each chunk (e.g. "doc1_chunk_0").
                ChromaDB will raise an error if you try to add a duplicate ID.
    metadatas : Dicts of arbitrary metadata per chunk (source filename, etc.)
    """
    collection = get_collection()
    collection.upsert(
        # upsert() inserts new IDs and overwrites existing ones
        documents=texts,
        ids=ids,
        metadatas=metadatas,
    )


def query_collection(query_text: str, n_results: int = TOP_K_RESULTS) -> list[RetrievedContext]:
    """
    Embed query_text and return the n_results most similar chunks.

    ChromaDB embeds the query, computes cosine distance against every
    stored vector, and returns the closest matches with their metadata.

    The raw ChromaDB response looks like:
        {
            "documents": [["chunk text 1", "chunk text 2", ...]],
            "metadatas": [[{"source": "file.txt", "chunk_index": 0}, ...]],
            "distances": [[0.12, 0.34, ...]],
        }

    Note the nested lists — ChromaDB supports batch queries, so the outer
    list is "one result set per query". We always send a single query here
    so we take index [0] from each list.

    Returns
    -------
    List of RetrievedContext objects, sorted best-match first.
    """
    collection = get_collection()

    # if the collection is empty, return nothing rather than crashing.
    if collection.count() == 0:
        return []

    # make sure we don't ask for more results than exist
    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query_text],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    # zip the parallel lists into typed RetrievedContext objects.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    return [
        RetrievedContext(
            text=doc,
            source=meta.get("source", "unknown"),
            chunk_index=int(meta.get("chunk_index", 0)),
            distance=dist,
        )
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def collection_count() -> int:
    """Return the total number of chunks currently stored in ChromaDB."""
    return get_collection().count()
