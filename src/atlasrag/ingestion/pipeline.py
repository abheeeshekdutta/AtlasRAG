from dataclasses import dataclass
from pathlib import Path

from atlasrag.ingestion.artifact_store import (
    save_chunked_document,
    save_embedded_document,
    save_parsed_document,
)
from atlasrag.ingestion.chunker import (
    CHUNKER_NAME,
    ChunkedDocument,
    DoclingHybridChunker,
)
from atlasrag.ingestion.embedder import (
    DocumentEmbedder,
    EmbeddedDocument,
    embed_chunked_document,
)
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser, ParsedDocument


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Result of ingesting one local document."""

    parsed: ParsedDocument
    artifact_path: Path
    chunked: ChunkedDocument
    chunk_artifact_path: Path
    embedded: EmbeddedDocument
    embedding_artifact_path: Path


def build_artifact_path(
    *,
    artifact_root: Path,
    parsed: ParsedDocument,
) -> Path:
    """Build a repeatable location for a parsed artifact."""
    document_id = parsed.version.source.document_id
    version_id = parsed.version.document_version_id

    filename = (
        f"parsed-v{parsed.schema_version}-"
        f"{parsed.parser_name}-{parsed.parser_version}.json"
    )

    return artifact_root / str(document_id) / str(version_id) / filename


def build_chunk_artifact_path(
    *,
    artifact_root: Path,
    chunked: ChunkedDocument,
) -> Path:
    """Build a repeatable location for a chunk artifact."""
    filename = (
        f"chunks-v{chunked.schema_version}-{CHUNKER_NAME}-"
        f"{chunked.chunker_fingerprint}.json"
    )
    return (
        artifact_root
        / str(chunked.version.source.document_id)
        / str(chunked.version.document_version_id)
        / filename
    )


def build_embedding_artifact_path(
    *,
    artifact_root: Path,
    embedded: EmbeddedDocument,
) -> Path:
    """Build a repeatable location for an embedding artifact."""
    filename = (
        f"embeddings-v{embedded.schema_version}-{embedded.embedder_fingerprint}.json"
    )
    return (
        artifact_root
        / str(embedded.version.source.document_id)
        / str(embedded.version.document_version_id)
        / filename
    )


def ingest_local_document(
    *,
    path: Path,
    source: DocumentSource,
    artifact_root: Path,
    parser: DoclingParser,
    chunker: DoclingHybridChunker,
    embedder: DocumentEmbedder,
) -> IngestionResult:
    """Load, parse, chunk, embed, and save one local document."""
    loaded = load_local_document(
        path=path,
        source=source,
    )
    parsed = parser.parse(loaded)

    artifact_path = build_artifact_path(
        artifact_root=artifact_root,
        parsed=parsed,
    )
    save_parsed_document(
        parsed,
        artifact_path,
    )

    chunked = chunker.chunk(parsed)
    chunk_artifact_path = build_chunk_artifact_path(
        artifact_root=artifact_root,
        chunked=chunked,
    )
    save_chunked_document(
        chunked,
        chunk_artifact_path,
    )

    embedded = embed_chunked_document(chunked, embedder)
    embedding_artifact_path = build_embedding_artifact_path(
        artifact_root=artifact_root,
        embedded=embedded,
    )
    save_embedded_document(embedded, embedding_artifact_path)

    return IngestionResult(
        parsed=parsed,
        artifact_path=artifact_path,
        chunked=chunked,
        chunk_artifact_path=chunk_artifact_path,
        embedded=embedded,
        embedding_artifact_path=embedding_artifact_path,
    )
