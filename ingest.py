"""
Document ingestion script — run this once before starting the API server.

Usage
─────
    python ingest.py                  # embed everything in ./data
    python ingest.py --data-dir docs  # use a different folder
    python ingest.py --clear          # wipe ChromaDB first, then re-embed

"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.embedder import add_chunks, collection_count, get_collection

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
SUPPORTED_EXTENSIONS = {".txt", ".md"}


# chunking

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into overlapping chunks, preferring natural break points.

    Strategy (tried in order):
      1. Paragraph breaks  (\\n\\n)
      2. Sentence endings  (. ! ? followed by whitespace)
      3. Word boundaries   (any whitespace)
      4. Hard character split (fallback — should be rare)

    Parameters
    ----------
    text         : The full document text.
    chunk_size   : Target maximum character length per chunk.
    chunk_overlap: How many characters of each chunk to repeat in the next.

    Returns
    -------
    List of chunk strings, each ≤ chunk_size characters (except when a
    single "atomic" unit like a very long sentence exceeds chunk_size).
    """
    if not text.strip():
        return []

    # normalise excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # priority-ordered separators
    separators = ["\n\n", r"(?<=[.!?])\s+", r"\s+", ""]

    def _split_with_sep(s: str, sep_pattern: str) -> list[str]:
        """Split string by a regex separator, keeping non-empty parts."""
        if sep_pattern == "":
            # hard character spli_t
            return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
        parts = re.split(sep_pattern, s)
        return [p for p in parts if p.strip()]

    def _merge_splits(splits: list[str]) -> list[str]:
        """
        Merge small splits back together into chunks of ≤ chunk_size,
        then add overlap between consecutive chunks.
        """
        chunks: list[str] = []
        current = ""

        for split in splits:
            candidate = (current + "\n\n" + split).strip() if current else split.strip()

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # start new chunk. carry over the overlap from the previous chunk.
                if chunk_overlap > 0 and current:
                    overlap_text = current[-chunk_overlap:]
                    current = (overlap_text + "\n\n" + split).strip()
                else:
                    current = split.strip()

        if current:
            chunks.append(current)

        return chunks

    for sep in separators:
        splits = _split_with_sep(text, sep)
        # if every split already fits in chunk_size, stop splitting
        if all(len(s) <= chunk_size for s in splits):
            return _merge_splits(splits)

    # absolute fallback
    return _merge_splits(_split_with_sep(text, ""))


# file reading

def _read_file(path: Path) -> str | None:
    """
    Read a file as UTF-8 text. Returns None if the file can't be decoded
    (e.g. accidentally pointed at a binary file).
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("Skipping %s — not valid UTF-8 text.", path.name)
        return None


# main ingestion logic

def ingest(data_dir: Path, chunk_size: int, chunk_overlap: int, clear: bool) -> None:
    """
    Read all supported files in data_dir, chunk them, and upsert into ChromaDB.

    Parameters
    ----------
    data_dir     : Directory containing source documents.
    chunk_size   : Maximum characters per chunk.
    chunk_overlap: Character overlap between consecutive chunks.
    clear        : If True, delete all existing vectors before ingesting.
    """
    if not data_dir.exists():
        logger.error("Data directory %r does not exist. Create it and add documents.", str(data_dir))
        sys.exit(1)

    # collect supported files
    files = [
        f for f in sorted(data_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.error(
            "No supported files found in %r. "
            "Add .txt or .md files and try again.",
            str(data_dir),
        )
        sys.exit(1)

    logger.info("Found %d file(s) in %r.", len(files), str(data_dir))

    # optionally wipe the collection before re-ingesting
    if clear:
        collection = get_collection()
        existing = collection.count()
        if existing > 0:
            # ChromaDB doesn't have a "clear" method — delete and recreate.
            # get_collection() uses get_or_create, so calling it again after
            # delete re-creates it fresh.
            from app.embedder import get_chroma_client, CHROMA_COLLECTION
            client = get_chroma_client()
            client.delete_collection(CHROMA_COLLECTION)
            logger.info("Cleared %d existing chunks from ChromaDB.", existing)
        else:
            logger.info("Collection was already empty — nothing to clear.")

    total_chunks = 0

    for file_path in files:
        logger.info("Processing: %s", file_path.name)

        text = _read_file(file_path)
        if text is None:
            continue

        if not text.strip():
            logger.warning("  Skipping %s — file is empty.", file_path.name)
            continue

        chunks = _split_text(text, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning("  No chunks produced from %s.", file_path.name)
            continue

        logger.info("  → %d chunk(s) from %s", len(chunks), file_path.name)

        # build the parallel lists that add_chunks expects.
        # ids must be unique across the entire collection.
        # format: "<filename>_chunk_<index>" e.g. "report_chunk_0"
        stem = file_path.stem  # filename without extension
        ids = [f"{stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file_path.name,   # e.g. "report.txt"
                "chunk_index": i,
                "chunk_size": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

        # add the chunks to the collection
        add_chunks(texts=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    final_count = collection_count()
    logger.info(
        "Done. Added %d new chunk(s). Total chunks in store: %d.",
        total_chunks,
        final_count,
    )


# cli entry point

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk and embed documents from a data directory into ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing .txt / .md source documents",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Character overlap between consecutive chunks",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing vectors before ingesting (full re-embed)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logger.info(
        "Ingestion config: data_dir=%r  chunk_size=%d  chunk_overlap=%d  clear=%s",
        str(args.data_dir),
        args.chunk_size,
        args.chunk_overlap,
        args.clear,
    )

    ingest(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clear=args.clear,
    )
