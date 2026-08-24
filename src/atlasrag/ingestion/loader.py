from dataclasses import dataclass
from pathlib import Path

from atlasrag.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    DocumentVersion,
)

_EXTENSION_TO_FORMAT = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
}


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Source bytes coupled to their derived version metadata."""

    path: Path
    content: bytes
    version: DocumentVersion


def detect_document_format(path: Path) -> DocumentFormat:
    """Detect a supported document format from its filename extension."""
    extension = path.suffix.casefold()

    try:
        return _EXTENSION_TO_FORMAT[extension]
    except KeyError as error:
        displayed_extension = extension or "<none>"
        raise ValueError(
            f"Unsupported document extension: {displayed_extension}"
        ) from error


def load_local_document(
    *,
    path: Path,
    source: DocumentSource,
) -> LoadedDocument:
    """Read a supported local document and derive its version metadata."""
    source_format = detect_document_format(path)
    content = path.read_bytes()

    if not content:
        raise ValueError(f"Document is empty: {path}")

    version = DocumentVersion.from_content(
        source=source,
        source_format=source_format,
        content=content,
    )

    return LoadedDocument(
        path=path,
        content=content,
        version=version,
    )
