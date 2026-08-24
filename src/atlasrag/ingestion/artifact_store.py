from pathlib import Path

from atlasrag.ingestion.parser import ParsedDocument


def save_parsed_document(
    parsed: ParsedDocument,
    destination: Path,
) -> None:
    """Save a parsed document using a temporary file."""
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_name(f".{destination.name}.tmp")

    try:
        temporary.write_text(
            parsed.model_dump_json(
                exclude_computed_fields=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_parsed_document(source: Path) -> ParsedDocument:
    """Load and validate a saved parsed document."""
    content = source.read_text(encoding="utf-8")
    return ParsedDocument.model_validate_json(content)
