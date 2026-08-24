from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.text import (
    SectionHeaderItem,
    TitleItem,
)

from atlasrag.ingestion.loader import load_local_document
from atlasrag.ingestion.models import DocumentFormat, DocumentSource
from atlasrag.ingestion.parser import (
    DoclingParser,
    DocumentParsingError,
    ParsedDocument,
)


def test_docling_parser_preserves_lineage_and_structure() -> None:
    path = Path("tests/fixtures/documents/security_policy.md")
    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="security/policy",
    )
    loaded = load_local_document(
        path=path,
        source=source,
    )

    parsed = DoclingParser().parse(loaded)

    assert parsed.version == loaded.version
    assert parsed.version.source_format is DocumentFormat.MARKDOWN
    assert parsed.parser_name == "docling"
    assert parsed.parser_version == distribution_version("docling")
    assert parsed.document.name == "security_policy"

    items = [item for item, _tree_level in parsed.document.iterate_items()]

    assert any(
        isinstance(item, TitleItem) and item.text == "Security Incident Policy"
        for item in items
    )

    headings = [
        (item.text, item.level) for item in items if isinstance(item, SectionHeaderItem)
    ]
    assert headings == [
        ("Reporting procedure", 1),
        ("Severity levels", 1),
        ("After-hours incidents", 2),
    ]

    tables = [item for item in items if isinstance(item, TableItem)]
    assert len(tables) == 1
    assert tables[0].data.num_rows == 3
    assert tables[0].data.num_cols == 3


def test_docling_parser_translates_conversion_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"this is not a valid PDF")

    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="invalid/pdf",
    )
    loaded = load_local_document(
        path=path,
        source=source,
    )

    with pytest.raises(
        DocumentParsingError,
        match="Docling conversion failed for invalid.pdf",
    ) as captured:
        DoclingParser().parse(loaded)

    assert captured.value.__cause__ is not None


def test_parsed_document_round_trips_through_json() -> None:
    path = Path("tests/fixtures/documents/security_policy.md")
    source = DocumentSource(
        source_namespace="test-fixtures",
        source_key="security/policy",
    )
    loaded = load_local_document(
        path=path,
        source=source,
    )
    original = DoclingParser().parse(loaded)

    serialized = original.model_dump_json(
        exclude_computed_fields=True,
    )
    restored = ParsedDocument.model_validate_json(serialized)

    assert restored.schema_version == 1
    assert restored.version == original.version
    assert restored.parser_name == original.parser_name
    assert restored.parser_version == original.parser_version
    assert restored.document.model_dump() == original.document.model_dump()
