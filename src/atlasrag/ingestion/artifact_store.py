from pathlib import Path

from atlasrag.ingestion.chunker import ChunkedDocument
from atlasrag.ingestion.embedder import EmbeddedDocument
from atlasrag.ingestion.parser import ParsedDocument


def _save_json(content: str, destination: Path) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_parsed_document(
    parsed: ParsedDocument,
    destination: Path,
) -> None:
    """Save a parsed document using a temporary file."""
    _save_json(
        parsed.model_dump_json(exclude_computed_fields=True),
        destination,
    )


def load_parsed_document(source: Path) -> ParsedDocument:
    """Load and validate a saved parsed document."""
    content = source.read_text(encoding="utf-8")
    return ParsedDocument.model_validate_json(content)


def save_chunked_document(
    chunked: ChunkedDocument,
    destination: Path,
) -> None:
    """Save a chunk artifact using a temporary file."""
    _save_json(
        chunked.model_dump_json(exclude_computed_fields=True),
        destination,
    )


def load_chunked_document(source: Path) -> ChunkedDocument:
    """Load and validate a saved chunk artifact."""
    content = source.read_text(encoding="utf-8")
    return ChunkedDocument.model_validate_json(content)


def save_embedded_document(
    embedded: EmbeddedDocument,
    destination: Path,
) -> None:
    """Save an embedding artifact using a temporary file."""
    _save_json(
        embedded.model_dump_json(exclude_computed_fields=True),
        destination,
    )


def load_embedded_document(source: Path) -> EmbeddedDocument:
    """Load and validate a saved embedding artifact."""
    content = source.read_text(encoding="utf-8")
    return EmbeddedDocument.model_validate_json(content)
