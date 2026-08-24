from dataclasses import dataclass
from pathlib import Path

from atlasrag.ingestion.artifact_store import save_parsed_document
from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentSource
from atlasrag.ingestion.parser import DoclingParser, ParsedDocument


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Result of ingesting one local document."""

    parsed: ParsedDocument
    artifact_path: Path


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


def ingest_local_document(
    *,
    path: Path,
    source: DocumentSource,
    artifact_root: Path,
    parser: DoclingParser,
) -> IngestionResult:
    """Load, parse, and save one local document."""
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

    return IngestionResult(
        parsed=parsed,
        artifact_path=artifact_path,
    )
