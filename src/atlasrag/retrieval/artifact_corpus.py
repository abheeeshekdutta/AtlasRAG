from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from atlasrag.ingestion.artifact_store import (
    load_chunked_document,
    load_embedded_document,
)
from atlasrag.ingestion.chunker import CHUNKER_NAME
from atlasrag.ingestion.embedder import QueryEmbedder, create_embedder_fingerprint
from atlasrag.retrieval.vector_index import InMemoryVectorIndex, VectorIndexError


class ArtifactCorpusError(RuntimeError):
    """Raised when persisted retrieval artifacts cannot form a valid corpus."""


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    """Hydrated vector index and counts for the artifacts loaded into it."""

    index: InMemoryVectorIndex
    document_count: int
    chunk_count: int


def load_artifact_corpus(
    *,
    artifact_root: Path,
    embedder: QueryEmbedder,
) -> LoadedCorpus:
    """Load compatible chunk and embedding artifacts into a fresh index."""
    if not artifact_root.exists():
        raise FileNotFoundError(f"Artifact root does not exist: {artifact_root}")
    if not artifact_root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {artifact_root}")

    embedder_fingerprint = create_embedder_fingerprint(
        embedder.config,
        provider_version=embedder.provider_version,
    )
    embedding_pattern = f"embeddings-v1-{embedder_fingerprint}.json"
    embedding_paths = sorted(artifact_root.glob(f"**/{embedding_pattern}"))

    index = InMemoryVectorIndex(embedder)
    document_count = 0
    chunk_count = 0

    for embedding_path in embedding_paths:
        try:
            embedded = load_embedded_document(embedding_path)
        except (OSError, ValidationError) as error:
            raise ArtifactCorpusError(
                f"Invalid embedding artifact: {embedding_path}"
            ) from error

        chunk_filename = f"chunks-v1-{CHUNKER_NAME}-{embedded.chunker_fingerprint}.json"
        chunk_path = embedding_path.with_name(chunk_filename)
        if not chunk_path.is_file():
            raise ArtifactCorpusError(
                f"Missing chunk artifact for embedding artifact: {embedding_path}"
            )

        try:
            chunked = load_chunked_document(chunk_path)
            index.add(chunked=chunked, embedded=embedded)
        except (OSError, ValidationError, VectorIndexError) as error:
            raise ArtifactCorpusError(
                f"Incompatible retrieval artifacts: {chunk_path} and {embedding_path}"
            ) from error

        document_count += 1
        chunk_count += len(chunked.chunks)

    return LoadedCorpus(
        index=index,
        document_count=document_count,
        chunk_count=chunk_count,
    )
